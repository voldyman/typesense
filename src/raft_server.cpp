#include "store.h"
#include "raft_server.h"
#include <butil/files/file_enumerator.h>
#include <thread>
#include <algorithm>
#include <string_utils.h>
#include <file_utils.h>
#include <collection_manager.h>
#include <http_client.h>
#include "rocksdb/utilities/checkpoint.h"

namespace braft {
    DECLARE_int32(raft_do_snapshot_min_index_gap);
}

void ReplicationClosure::Run() {
    // nothing much to do here since responding to client is handled upstream
    // Auto delete `this` after Run()
    std::unique_ptr<ReplicationClosure> self_guard(this);
}

// State machine implementation

int ReplicationState::start(const butil::EndPoint & peering_endpoint, const int api_port,
                            int election_timeout_ms, int snapshot_interval_s,
                            const std::string & raft_dir, const std::string & nodes) {

    this->election_timeout_interval_ms = election_timeout_ms;
    this->raft_dir_path = raft_dir;

    braft::NodeOptions node_options;
    std::string actual_nodes_config = to_nodes_config(peering_endpoint, api_port, nodes);

    if(node_options.initial_conf.parse_from(actual_nodes_config) != 0) {
        LOG(ERROR) << "Failed to parse nodes configuration `" << nodes << "`";
        return -1;
    }

    this->caught_up = false;

    // do snapshot only when the gap between applied index and last snapshot index is >= this number
    braft::FLAGS_raft_do_snapshot_min_index_gap = 1;

    node_options.election_timeout_ms = election_timeout_ms;
    node_options.fsm = this;
    node_options.node_owns_fsm = false;
    node_options.snapshot_interval_s = snapshot_interval_s;
    node_options.filter_before_copy_remote = true;
    std::string prefix = "local://" + raft_dir;
    node_options.log_uri = prefix + "/" + log_dir_name;
    node_options.raft_meta_uri = prefix + "/" + meta_dir_name;
    node_options.snapshot_uri = prefix + "/" + snapshot_dir_name;
    node_options.disable_cli = true;

    // api_port is used as the node identifier
    braft::Node* node = new braft::Node("default_group", braft::PeerId(peering_endpoint, api_port));

    std::string snapshot_dir = raft_dir + "/" + snapshot_dir_name;
    bool snapshot_exists = dir_enum_count(snapshot_dir) > 0;

    if(snapshot_exists) {
        // we will be assured of on_snapshot_load() firing and we will wait for that to init_db()
    } else if(!create_init_db_snapshot) {
        // `create_init_db_snapshot` can be handled separately only after leader starts
        LOG(INFO) << "Snapshot does not exist. We will remove db dir and init db fresh.";

        reset_db();
        if (!butil::DeleteFile(butil::FilePath(store->get_state_dir_path()), true)) {
            LOG(WARNING) << "rm " << store->get_state_dir_path() << " failed";
            return -1;
        }

        int init_db_status = init_db();
        if(init_db_status != 0) {
            LOG(ERROR) << "Failed to initialize DB.";
            return init_db_status;
        }
    }

    if (node->init(node_options) != 0) {
        LOG(ERROR) << "Fail to init peering node";
        delete node;
        return -1;
    }

    std::vector<std::string> peer_vec;
    StringUtils::split(actual_nodes_config, peer_vec, ",");

    this->node = node;
    return 0;
}

std::string ReplicationState::to_nodes_config(const butil::EndPoint& peering_endpoint, const int api_port,
                                              const std::string& nodes_config) {
    std::string actual_nodes_config = nodes_config;

    if(nodes_config.empty()) {
        std::string ip_str = butil::ip2str(peering_endpoint.ip).c_str();
        actual_nodes_config = ip_str + ":" + std::to_string(peering_endpoint.port) + ":" + std::to_string(api_port);
    }

    return actual_nodes_config;
}

void ReplicationState::write(http_req* request, http_res* response) {
    if(!node) {
        return ;
    }

    if (!node->is_leader()) {
        return write_to_leader(request, response);
    }

    // Serialize request to replicated WAL so that all the nodes in the group receive it as well.
    // NOTE: actual write must be done only on the `on_apply` method to maintain consistency.

    butil::IOBufBuilder bufBuilder;
    bufBuilder << request->serialize();

    // Apply this log as a braft::Task

    braft::Task task;
    task.data = &bufBuilder.buf();
    // This callback would be invoked when the task actually executes or fails
    task.done = new ReplicationClosure(request, response);

    // To avoid ABA problem
    task.expected_term = leader_term.load(butil::memory_order_relaxed);

    //LOG(INFO) << ":::" << "body size before apply: " << request->body.size();

    // Now the task is applied to the group, waiting for the result.
    return node->apply(task);
}

void ReplicationState::write_to_leader(http_req *request, http_res *response) const {
    if(node->leader_id().is_empty()) {
        // Handle no leader scenario
        LOG(ERROR) << "Rejecting write: could not find a leader.";

        if(request->_req->proceed_req && response->proxied_stream) {
            // streaming in progress: ensure graceful termination (cannot start response again)
            LOG(ERROR) << "Terminating streaming request gracefully.";
            request->await.notify();
            return ;
        }

        response->set_500("Could not find a leader.");
        auto replication_arg = new AsyncIndexArg{request, response, nullptr};
        replication_arg->req->route_hash = static_cast<uint64_t>(ROUTE_CODES::ALREADY_HANDLED);
        return message_dispatcher->send_message(REPLICATION_MSG, replication_arg);
    }

    if (request->_req->proceed_req && response->proxied_stream) {
        // indicates async request body of in-flight request
        //LOG(INFO) << "Inflight proxied request, returning control to caller, body_size=" << request->body.size();
        request->await.notify();
        return ;
    }

    const std::string & leader_addr = node->leader_id().to_string();
    LOG(INFO) << "Redirecting write to leader at: " << leader_addr;

    h2o_custom_generator_t* custom_generator = reinterpret_cast<h2o_custom_generator_t *>(response->generator);
    HttpServer* server = custom_generator->h2o_handler->http_server;

    thread_pool->enqueue([leader_addr, request, response, server, this]() {
        auto raw_req = request->_req;
        const std::string& path = std::string(raw_req->path.base, raw_req->path.len);
        const std::string& scheme = std::string(raw_req->scheme->name.base, raw_req->scheme->name.len);
        const std::string url = get_leader_url_path(leader_addr, path, scheme);

        std::map<std::string, std::string> res_headers;

        if(request->http_method == "POST") {
            std::vector<std::string> path_parts;
            StringUtils::split(path, path_parts, "/");

            if(path_parts.back().rfind("import", 0) == 0) {
                // imports are handled asynchronously
                response->proxied_stream = true;
                response->auto_dispose = false;
                long status = HttpClient::post_response_async(url, request, response, server);

                // must manage life cycle for forwarded requests
                delete request;
                delete response;

                //LOG(INFO) << "Import call done.";

                if(status == 500) {
                    response->content_type_header = res_headers["content-type"];
                    response->set_500("");
                } else {
                    return ;
                }
            } else {
                std::string api_res;
                long status = HttpClient::post_response(url, request->body, api_res, res_headers);
                response->content_type_header = res_headers["content-type"];
                response->set_body(status, api_res);
            }
        } else if(request->http_method == "PUT") {
            std::string api_res;
            long status = HttpClient::put_response(url, request->body, api_res, res_headers);
            response->content_type_header = res_headers["content-type"];
            response->set_body(status, api_res);
        } else if(request->http_method == "DELETE") {
            std::string api_res;
            long status = HttpClient::delete_response(url, api_res, res_headers);
            response->content_type_header = res_headers["content-type"];
            response->set_body(status, api_res);
        } else {
            const std::string& err = "Forwarding for http method not implemented: " + request->http_method;
            LOG(ERROR) << err;
            response->set_500(err);
        }

        auto replication_arg = new AsyncIndexArg{request, response, nullptr};
        replication_arg->req->route_hash = static_cast<uint64_t>(ROUTE_CODES::ALREADY_HANDLED);
        message_dispatcher->send_message(REPLICATION_MSG, replication_arg);
    });
}

std::string ReplicationState::get_leader_url_path(const std::string& leader_addr, const std::string& path,
                                                  const std::string& protocol) const {
    std::vector<std::string> addr_parts;
    StringUtils::split(leader_addr, addr_parts, ":");
    std::string leader_host_port = addr_parts[0] + ":" + addr_parts[2];
    std::string url = protocol + "://" + leader_host_port + path;
    return url;
}

void ReplicationState::on_apply(braft::Iterator& iter) {
    //LOG(INFO) << "ReplicationState::on_apply";

    // NOTE: this is executed on a different thread and runs concurrent to http thread
    // A batch of tasks are committed, which must be processed through
    // |iter|
    for (; iter.valid(); iter.next()) {
        http_res* response;
        http_req* request;

        // Guard invokes replication_arg->done->Run() asynchronously to avoid the callback blocking the main thread
        braft::AsyncClosureGuard closure_guard(iter.done());

        if (iter.done()) {
            // This task is applied by this node, get value from the closure to avoid additional parsing.
            ReplicationClosure* c = dynamic_cast<ReplicationClosure*>(iter.done());
            response = c->get_response();
            request = c->get_request();

            //LOG(INFO) << ":::" << "body size inside apply: " << request->body.size();
        } else {
            // Parse request from the log
            response = new http_res;

            request = new http_req;
            request->deserialize(iter.data().to_string());

            //LOG(INFO) << "Parsed request from the log, body_size: " << request->body.size();
        }

        if(request->_req == nullptr && request->body == "INIT_SNAPSHOT") {
            // We attempt to trigger a cold snapshot against an existing stand-alone DB for backward compatibility
            InitSnapshotClosure* init_snapshot_closure = new InitSnapshotClosure(this);
            node->snapshot(init_snapshot_closure);
            delete request;
            delete response;
            continue ;
        }

        // Now that the log has been parsed, perform the actual operation
        // Call http server thread for write and response back to client (if `response` is NOT null)
        // We use a future to block current thread until the async flow finishes
        response->auto_dispose = false;
        auto replication_arg = new AsyncIndexArg{request, response, nullptr};
        message_dispatcher->send_message(REPLICATION_MSG, replication_arg);

        //LOG(INFO) << "Raft write waiting to proceed";
        response->await.wait();
        //LOG(INFO) << "Raft write ready to proceed, response->final=" << response->final;

        if(response->final) {
            delete request;
            delete response;
        }

        if(shut_down) {
            iter.set_error_and_rollback();
            return;
        }
    }
}

void ReplicationState::read(http_res* response) {
    // NOT USED:
    // For consistency, reads to followers could be rejected.
    // Currently, we don't do implement reads via raft.
}

void* ReplicationState::save_snapshot(void* arg) {
    LOG(INFO) << "save_snapshot called";

    SnapshotArg* sa = static_cast<SnapshotArg*>(arg);
    std::unique_ptr<SnapshotArg> arg_guard(sa);
    brpc::ClosureGuard done_guard(sa->done);

    // add the db snapshot files to writer state
    butil::FileEnumerator dir_enum(butil::FilePath(sa->db_snapshot_path), false, butil::FileEnumerator::FILES);

    for (butil::FilePath file = dir_enum.Next(); !file.empty(); file = dir_enum.Next()) {
        std::string file_name = std::string(db_snapshot_name) + "/" + file.BaseName().value();
        if (sa->writer->add_file(file_name) != 0) {
            sa->done->status().set_error(EIO, "Fail to add file to writer.");
            return nullptr;
        }
    }

    // if an external snapshot is requested, copy both state and data directory into that
    if(!sa->ext_snapshot_path.empty()) {
        LOG(INFO) << "Copying system snapshot to external snapshot directory at " << sa->ext_snapshot_path;
        if(!butil::DirectoryExists(butil::FilePath(sa->ext_snapshot_path))) {
            butil::CreateDirectory(butil::FilePath(sa->ext_snapshot_path), true);
        }
        butil::CopyDirectory(butil::FilePath(sa->state_dir_path), butil::FilePath(sa->ext_snapshot_path), true);
        butil::CopyDirectory(butil::FilePath(sa->db_dir_path), butil::FilePath(sa->ext_snapshot_path), true);
    }

    // NOTE: *must* do a dummy write here since snapshots cannot be triggered if no write has happened since the
    // last snapshot. By doing a dummy write right after a snapshot, we ensure that this can never be the case.
    sa->replication_state->do_dummy_write();

    LOG(INFO) << "save_snapshot done";

    return nullptr;
}

// this method is serial to on_apply so guarantees a snapshot view of the state machine
void ReplicationState::on_snapshot_save(braft::SnapshotWriter* writer, braft::Closure* done) {
    LOG(INFO) << "on_snapshot_save";

    rocksdb::Checkpoint* checkpoint = nullptr;
    rocksdb::Status status = rocksdb::Checkpoint::Create(store->_get_db_unsafe(), &checkpoint);

    if(!status.ok()) {
        LOG(WARNING) << "Checkpoint Create failed, msg:" << status.ToString();
        done->status().set_error(EIO, "Checkpoint Create failed.");
    }

    std::string db_snapshot_path = writer->get_path() + "/" + db_snapshot_name;

    std::unique_ptr<rocksdb::Checkpoint> checkpoint_guard(checkpoint);
    status = checkpoint->CreateCheckpoint(db_snapshot_path);

    if(!status.ok()) {
        LOG(WARNING) << "Checkpoint CreateCheckpoint failed at snapshot path: "
                     << db_snapshot_path << ", msg:" << status.ToString();
        done->status().set_error(EIO, "CreateCheckpoint failed.");
    }

    SnapshotArg* arg = new SnapshotArg;
    arg->replication_state = this;
    arg->writer = writer;
    arg->state_dir_path = raft_dir_path;
    arg->db_dir_path = store->get_state_dir_path();
    arg->db_snapshot_path = db_snapshot_path;
    arg->done = done;

    if(!ext_snapshot_path.empty()) {
        arg->ext_snapshot_path = ext_snapshot_path;
        ext_snapshot_path = "";
    }

    // Start a new bthread to avoid blocking StateMachine since it could be slow to write data to disk
    bthread_t tid;
    bthread_start_urgent(&tid, NULL, save_snapshot, arg);
}

int ReplicationState::init_db() {
    if (!butil::CreateDirectory(butil::FilePath(store->get_state_dir_path()))) {
        LOG(WARNING) << "CreateDirectory " << store->get_state_dir_path() << " failed";
        return -1;
    }

    const rocksdb::Status& status = store->init_db();
    if (!status.ok()) {
        LOG(WARNING) << "Open DB " << store->get_state_dir_path() << " failed, msg: " << status.ToString();
        return -1;
    }

    LOG(INFO) << "DB open success!";
    LOG(INFO) << "Loading collections from disk...";

    Option<bool> init_op = CollectionManager::get_instance().load();

    if(init_op.ok()) {
        LOG(INFO) << "Finished loading collections from disk.";
    } else {
        LOG(ERROR)<< "Typesense failed to start. " << "Could not load collections from disk: " << init_op.error();
        return 1;
    }

    return 0;
}

int ReplicationState::on_snapshot_load(braft::SnapshotReader* reader) {
    CHECK(!node || !node->is_leader()) << "Leader is not supposed to load snapshot";

    LOG(INFO) << "on_snapshot_load";

    // Load snapshot from reader, replacing the running StateMachine

    reset_db();
    if (!butil::DeleteFile(butil::FilePath(store->get_state_dir_path()), true)) {
        LOG(WARNING) << "rm " << store->get_state_dir_path() << " failed";
        return -1;
    }

    LOG(INFO) << "rm " << store->get_state_dir_path() << " success";

    std::string snapshot_path = reader->get_path();
    snapshot_path.append(std::string("/") + db_snapshot_name);

    // tries to use link if possible, or else copies
    if (!copy_dir(snapshot_path, store->get_state_dir_path())) {
        LOG(WARNING) << "copy snapshot " << snapshot_path << " to " << store->get_state_dir_path() << " failed";
        return -1;
    }

    LOG(INFO) << "copy snapshot " << snapshot_path << " to " << store->get_state_dir_path() << " success";

    return init_db();
}

void ReplicationState::refresh_nodes(const std::string & nodes) {
    if(!node) {
        LOG(WARNING) << "Node state is not initialized: unable to refresh nodes.";
        return ;
    }

    braft::Configuration new_conf;
    new_conf.parse_from(nodes);

    if(node->is_leader()) {
        RefreshNodesClosure* refresh_nodes_done = new RefreshNodesClosure;
        node->change_peers(new_conf, refresh_nodes_done);
        this->caught_up = true;
    } else {
        if(node->leader_id().is_empty()) {
            // When node is not a leader, does not have a leader and is also a single-node cluster,
            // we forcefully reset its peers.
            // NOTE: `reset_peers()` is not a safe call to make as we give up on consistency and consensus guarantees.
            // We are doing this solely to handle single node cluster whose IP changes.
            // Examples: Docker container IP change, local DHCP leased IP change etc.

            std::vector<braft::PeerId> latest_nodes;
            new_conf.list_peers(&latest_nodes);

            if(latest_nodes.size() == 1) {
                LOG(WARNING) << "Single-node with no leader. Resetting peers.";
                node->reset_peers(new_conf);
            } else {
                LOG(WARNING) << "Multi-node with no leader: refusing to reset peers.";
            }

            this->caught_up = false;
            return ;
        }

        // update catch up status
        thread_pool->enqueue([this]() {
            auto seq_num = this->store->get_latest_seq_number();
            const std::string & leader_addr = node->leader_id().to_string();

            std::vector<std::string> addr_parts;
            StringUtils::split(leader_addr, addr_parts, ":");
            std::string leader_host_port = addr_parts[0] + ":" + addr_parts[2];
            std::string protocol = this->api_uses_ssl ? "https" : "http";
            std::string seq_url = protocol + "://" + leader_host_port + "/sequence";

            std::string api_res;
            std::map<std::string, std::string> res_headers;
            long status = HttpClient::get_response(seq_url, api_res, res_headers);

            if(status != 500) {
                if(!StringUtils::is_uint64_t(api_res)) {
                    LOG(ERROR) << "Invalid API response when fetching sequence number: " << api_res;
                    this->caught_up = false;
                    return ;
                }

                uint64_t leader_seq = std::atoll(api_res.c_str());
                if(leader_seq < seq_num) {
                    LOG(ERROR) << "Leader sequence " << leader_seq << " is less than local sequence " << seq_num;
                    this->caught_up = false;
                    return ;
                }

                const uint64_t seq_diff = leader_seq - seq_num;

                if(seq_diff < CATCHUP_MIN_SEQUENCE_DIFF) {
                    this->caught_up = true;
                    return ;
                }

                float seq_progress = (float(seq_num) / leader_seq) * 100;
                LOG(INFO) << "Follower progress percentage: " << seq_progress;
                this->caught_up = (seq_progress >= catch_up_threshold_percentage);
            }
        });
    }
}

ReplicationState::ReplicationState(Store *store, ThreadPool* thread_pool, http_message_dispatcher *message_dispatcher,
                                   bool api_uses_ssl, size_t catch_up_threshold_percentage,
                                   bool create_init_db_snapshot, std::atomic<bool>& quit_service):
        node(nullptr), leader_term(-1), store(store), thread_pool(thread_pool),
        message_dispatcher(message_dispatcher), catch_up_threshold_percentage(catch_up_threshold_percentage),
        api_uses_ssl(api_uses_ssl), create_init_db_snapshot(create_init_db_snapshot), shut_down(quit_service) {

}

void ReplicationState::reset_db() {
    store->close();
}

bool ReplicationState::is_alive() const {
    if(node == nullptr || !is_ready()) {
        return false;
    }

    // node should either be a leader or have a leader
    return (node->is_leader() || !node->leader_id().is_empty());
}

uint64_t ReplicationState::node_state() const {
    if(node == nullptr) {
        return 0;
    }

    braft::NodeStatus node_status;
    node->get_status(&node_status);

    return node_status.state;
}

void ReplicationState::do_snapshot(const std::string& snapshot_path, http_req& req, http_res& res) {
    LOG(INFO) << "Triggerring an on demand snapshot...";
    OnDemandSnapshotClosure* snapshot_closure = new OnDemandSnapshotClosure(this, req, res);
    ext_snapshot_path = snapshot_path;
    node->snapshot(snapshot_closure);
}

void ReplicationState::set_ext_snapshot_path(const std::string& snapshot_path) {
    this->ext_snapshot_path = snapshot_path;
}

const std::string &ReplicationState::get_ext_snapshot_path() const {
    return ext_snapshot_path;
}

void ReplicationState::do_dummy_write() {
    if(node->leader_id().is_empty()) {
        LOG(ERROR) << "Could not do a dummy write, as node does not have a leader";
        return ;
    }

    const std::string & leader_addr = node->leader_id().to_string();
    const std::string protocol = api_uses_ssl ? "https" : "http";
    std::string url = get_leader_url_path(leader_addr, "/health", protocol);

    std::string api_res;
    std::map<std::string, std::string> res_headers;
    long status_code = HttpClient::post_response(url, "", api_res, res_headers);

    LOG(INFO) << "Dummy write to " << url << ", status = " << status_code << ", response = " << api_res;
}

bool ReplicationState::trigger_vote() {
    if(node) {
        auto status = node->vote(election_timeout_interval_ms);
        LOG(INFO) << "Triggered vote. Ok? " << status.ok() << ", status: " << status;
        return status.ok();
    }

    return false;
}

void InitSnapshotClosure::Run() {
    // Auto delete this after Run()
    std::unique_ptr<InitSnapshotClosure> self_guard(this);

    if(status().ok()) {
        LOG(INFO) << "Init snapshot succeeded!";
        replication_state->reset_db();
        replication_state->init_db();
    } else {
        LOG(ERROR) << "Init snapshot failed, error: " << status().error_str() << ", code: " << status().error_code();
    }
}

void OnDemandSnapshotClosure::Run() {
    // Auto delete this after Run()
    std::unique_ptr<OnDemandSnapshotClosure> self_guard(this);

    replication_state->set_ext_snapshot_path("");

    req.last_chunk_aggregate = true;
    res.final = true;

    nlohmann::json response;
    uint32_t status_code;

    if(status().ok()) {
        LOG(INFO) << "On demand snapshot succeeded!";
        status_code = 201;
        response["success"] = true;
    } else {
        LOG(ERROR) << "On demand snapshot failed, error: " << status().error_str() << ", code: " << status().error_code();
        status_code = 500;
        response["success"] = false;
        response["error"] = status().error_str();
    }

    res.status_code = status_code;
    res.body = response.dump();

    HttpServer::stream_response(req, res);
}
