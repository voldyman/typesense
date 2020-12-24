#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <art.h>
#include <number.h>
#include <sparsepp.h>
#include <store.h>
#include <topster.h>
#include <json.hpp>
#include <field.h>
#include <option.h>
#include <set>
#include "string_utils.h"
#include "num_tree.h"

struct token_candidates {
    std::string token;
    size_t cost;
    std::vector<art_leaf*> candidates;
};

struct search_field_t {
    std::string name;
    size_t weight;
};

struct search_args {
    std::vector<std::string> q_include_tokens;
    std::vector<std::string> q_exclude_tokens;
    std::vector<std::vector<std::string>> q_synonyms;
    std::vector<search_field_t> search_fields;
    std::vector<filter> filters;
    std::vector<facet> facets;
    std::map<size_t, std::map<size_t, uint32_t>> included_ids;
    std::vector<uint32_t> excluded_ids;
    std::vector<sort_by> sort_fields_std;
    facet_query_t facet_query;
    int num_typos;
    size_t max_facet_values;
    size_t per_page;
    size_t page;
    token_ordering token_order;
    bool prefix;
    size_t drop_tokens_threshold;
    size_t typo_tokens_threshold;
    std::vector<std::string> group_by_fields;
    size_t group_limit;
    size_t all_result_ids_len;
    spp::sparse_hash_set<uint64_t> groups_processed;
    std::vector<std::vector<art_leaf*>> searched_queries;
    Topster* topster;
    Topster* curated_topster;
    std::vector<std::vector<KV*>> raw_result_kvs;
    std::vector<std::vector<KV*>> override_result_kvs;
    Option<uint32_t> outcome;

    search_args(): outcome(0) {

    }

    search_args(std::vector<std::string> q_include_tokens,
                std::vector<std::string> q_exclude_tokens,
                std::vector<std::vector<std::string>> q_synonyms,
                std::vector<search_field_t> search_fields, std::vector<filter> filters,
                std::vector<facet> facets, std::map<size_t, std::map<size_t, uint32_t>> included_ids, std::vector<uint32_t> excluded_ids,
                std::vector<sort_by> sort_fields_std, facet_query_t facet_query, int num_typos, size_t max_facet_values,
                size_t max_hits, size_t per_page, size_t page, token_ordering token_order, bool prefix,
                size_t drop_tokens_threshold, size_t typo_tokens_threshold,
                const std::vector<std::string>& group_by_fields, size_t group_limit):
            q_include_tokens(q_include_tokens), q_exclude_tokens(q_exclude_tokens), q_synonyms(q_synonyms),
            search_fields(search_fields), filters(filters), facets(facets),
            included_ids(included_ids), excluded_ids(excluded_ids), sort_fields_std(sort_fields_std),
            facet_query(facet_query), num_typos(num_typos), max_facet_values(max_facet_values), per_page(per_page),
            page(page), token_order(token_order), prefix(prefix),
            drop_tokens_threshold(drop_tokens_threshold), typo_tokens_threshold(typo_tokens_threshold),
            group_by_fields(group_by_fields), group_limit(group_limit),
            all_result_ids_len(0), outcome(0) {

        const size_t topster_size = std::max((size_t)1, max_hits);  // needs to be atleast 1 since scoring is mandatory
        topster = new Topster(topster_size, group_limit);
        curated_topster = new Topster(topster_size, group_limit);
    }

    ~search_args() {
        delete topster;
        delete curated_topster;
    };
};

enum index_operation_t {
    CREATE,
    UPSERT,
    UPDATE,
    DELETE
};


struct index_record {
    size_t position;                    // position of record in the original request
    uint32_t seq_id;

    nlohmann::json doc;
    nlohmann::json old_doc;
    nlohmann::json new_doc;
    nlohmann::json del_doc;

    index_operation_t operation;
    bool is_update;

    Option<bool> indexed;               // indicates if the indexing operation was a success

    index_record(size_t record_pos, uint32_t seq_id, const nlohmann::json& doc, index_operation_t operation):
            position(record_pos), seq_id(seq_id), doc(doc), operation(operation), is_update(false), indexed(false) {

    }

    void index_failure(const uint32_t err_code, const std::string & err_msg) {
        indexed = Option<bool>(err_code, err_msg);
    }

    void index_success() {
        indexed = Option<bool>(true);
    }
};

class Index {
private:
    const uint64_t FACET_ARRAY_DELIMETER = std::numeric_limits<uint64_t>::max();

    std::string name;

    size_t num_documents;

    std::unordered_map<std::string, field> search_schema;

    std::map<std::string, field> facet_schema;  // std::map guarantees order of fields

    std::unordered_map<std::string, field> sort_schema;

    spp::sparse_hash_map<std::string, art_tree*> search_index;

    spp::sparse_hash_map<std::string, num_tree_t*> numerical_index;

    // seq_id => (facet => values)
    spp::sparse_hash_map<uint32_t, std::vector<std::vector<uint64_t>>> facet_index_v2;

    // sort_field => (seq_id => value)
    spp::sparse_hash_map<std::string, spp::sparse_hash_map<uint32_t, int64_t>*> sort_index;

    StringUtils string_utils;

    static inline uint32_t next_suggestion(const std::vector<token_candidates> &token_candidates_vec,
                                       long long int n,
                                       std::vector<art_leaf *>& actual_query_suggestion,
                                       std::vector<art_leaf *>& query_suggestion);

    void log_leaves(const int cost, const std::string &token, const std::vector<art_leaf *> &leaves) const;

    void do_facets(std::vector<facet> & facets, facet_query_t & facet_query,
                   const uint32_t* result_ids, size_t results_size);

    void search_field(const uint8_t & field_id,
                      std::vector<std::string>& query_tokens,
                      std::vector<std::string>& search_tokens,
                      const uint32_t* exclude_token_ids,
                      size_t exclude_token_ids_size,
                      size_t& num_tokens_dropped,
                      const std::string & field, uint32_t *filter_ids, size_t filter_ids_length,
                      const std::vector<uint32_t>& curated_ids,
                      std::vector<facet> & facets, const std::vector<sort_by> & sort_fields,
                      const int num_typos, std::vector<std::vector<art_leaf*>> & searched_queries,
                      Topster* topster, spp::sparse_hash_set<uint64_t>& groups_processed,
                      uint32_t** all_result_ids, size_t & all_result_ids_len,
                      const token_ordering token_order = FREQUENCY, const bool prefix = false,
                      const size_t drop_tokens_threshold = Index::DROP_TOKENS_THRESHOLD,
                      const size_t typo_tokens_threshold = Index::TYPO_TOKENS_THRESHOLD);

    void search_candidates(const uint8_t & field_id,
                           uint32_t* filter_ids, size_t filter_ids_length,
                           const uint32_t* exclude_token_ids, size_t exclude_token_ids_size,
                           const std::vector<uint32_t>& curated_ids,
                           const std::vector<sort_by> & sort_fields, std::vector<token_candidates> & token_to_candidates,
                           std::vector<std::vector<art_leaf*>> & searched_queries,
                           Topster* topster, spp::sparse_hash_set<uint64_t>& groups_processed,
                           uint32_t** all_result_ids,
                           size_t & all_result_ids_len,
                           size_t& field_num_results,
                           const size_t typo_tokens_threshold);

    void insert_doc(const int64_t score, art_tree *t, uint32_t seq_id,
                    const std::unordered_map<std::string, std::vector<uint32_t>> &token_to_offsets) const;

    void index_string_field(const std::string & text, const int64_t score, art_tree *t, uint32_t seq_id,
                            int facet_id, const field & a_field);

    void index_string_array_field(const std::vector<std::string> & strings, const int64_t score, art_tree *t,
                                  uint32_t seq_id, int facet_id, const field & a_field);

    void remove_and_shift_offset_index(sorted_array& offset_index, const uint32_t* indices_sorted,
                                       const uint32_t indices_length);

    void collate_included_ids(const std::vector<std::string>& q_included_tokens,
                              const std::string & field, const uint8_t field_id,
                              const std::map<size_t, std::map<size_t, uint32_t>> & included_ids_map,
                              Topster* curated_topster, std::vector<std::vector<art_leaf*>> & searched_queries);

    uint64_t facet_token_hash(const field & a_field, const std::string &token);

    void compute_facet_stats(facet &a_facet, uint64_t raw_value, const std::string & field_type);

public:
    Index() = delete;

    Index(const std::string name, const std::unordered_map<std::string, field> & search_schema,
          std::map<std::string, field> facet_schema, std::unordered_map<std::string, field> sort_schema);

    ~Index();

    // reference: https://stackoverflow.com/a/27952689/131050
    static uint64_t hash_combine(uint64_t combined, uint64_t hash) {
        combined ^= hash + 0x517cc1b727220a95 + (combined << 6) + (combined >> 2);
        return combined;
    }

    void run_search();

    void search(Option<uint32_t> & outcome,
                          const std::vector<std::string> & q_include_tokens,
                          const std::vector<std::string> & q_exclude_tokens,
                          const std::vector<std::vector<std::string>>& q_synonyms,
                          const std::vector<search_field_t> & search_fields,
                          const std::vector<filter> & filters, std::vector<facet> & facets,
                          facet_query_t & facet_query,
                          const std::map<size_t, std::map<size_t, uint32_t>> & included_ids_map,
                          const std::vector<uint32_t> & excluded_ids,
                          const std::vector<sort_by> & sort_fields_std, const int num_typos,
                          Topster* topster, Topster* curated_topster,
                          const size_t per_page, const size_t page, const token_ordering token_order,
                          const bool prefix, const size_t drop_tokens_threshold,
                          size_t & all_result_ids_len,
                          spp::sparse_hash_set<uint64_t>& groups_processed,
                          std::vector<std::vector<art_leaf*>> & searched_queries,
                          std::vector<std::vector<KV*>> & raw_result_kvs,
                          std::vector<std::vector<KV*>> & override_result_kvs,
                          const size_t typo_tokens_threshold);

    static void concat_topster_ids(Topster* topster, spp::sparse_hash_map<uint64_t, std::vector<KV*>>& topster_ids);

    Option<uint32_t> do_filtering(uint32_t** filter_ids_out, const std::vector<filter> & filters);

    Option<uint32_t> remove(const uint32_t seq_id, const nlohmann::json & document);

    art_leaf* get_token_leaf(const std::string & field_name, const unsigned char* token, uint32_t token_len);

    static void populate_token_positions(const std::vector<art_leaf *> &query_suggestion,
                                         std::vector<uint32_t*>& leaf_to_indices,
                                         size_t result_index,
                                         std::unordered_map<size_t, std::vector<std::vector<uint16_t>>> &array_token_positions);

    void score_results(const std::vector<sort_by> & sort_fields, const uint16_t & query_index, const uint8_t & field_id,
                       const uint32_t total_cost, Topster* topster, const std::vector<art_leaf *> & query_suggestion,
                       spp::sparse_hash_set<uint64_t>& groups_processed,
                       const uint32_t *result_ids, const size_t result_size);

    static int64_t get_points_from_doc(const nlohmann::json &document, const std::string & default_sorting_field);

    Option<uint32_t> index_in_memory(const nlohmann::json & document, uint32_t seq_id,
                                     const std::string & default_sorting_field, bool is_update);

    static Option<uint32_t> validate_index_in_memory(const nlohmann::json &document, uint32_t seq_id,
                                                     const std::string & default_sorting_field,
                                                     const std::unordered_map<std::string, field> & search_schema,
                                                     const std::map<std::string, field> & facet_schema,
                                                     bool is_update);

    static size_t batch_memory_index(Index *index,
                                     std::vector<index_record> & iter_batch,
                                     const std::string & default_sorting_field,
                                     const std::unordered_map<std::string, field> & search_schema,
                                     const std::map<std::string, field> & facet_schema);

    const spp::sparse_hash_map<std::string, art_tree *> &_get_search_index() const;

    const spp::sparse_hash_map<std::string, num_tree_t*>& _get_numerical_index() const;

    // for limiting number of results on multiple candidates / query rewrites
    enum {TYPO_TOKENS_THRESHOLD = 100};

    // for limiting number of fields that can be searched on
    enum {FIELD_LIMIT_NUM = 100};

    // If the number of results found is less than this threshold, Typesense will attempt to drop the tokens
    // in the query that have the least individual hits one by one until enough results are found.
    static const int DROP_TOKENS_THRESHOLD = 10;

    /*
     * Concurrency Primitives
    */

    // Used for passing control back and forth between main and worker threads
    std::mutex m;
    std::condition_variable cv;

    bool ready;       // prevents spurious wake up of the worker thread
    bool processed;   // prevents spurious wake up of the main thread
    bool terminate;   // used for interrupting the thread during tear down

    search_args* search_params;

    int get_bounded_typo_cost(const size_t max_cost, const size_t token_len) const;

    static int64_t float_to_in64_t(float n);

    uint64_t get_distinct_id(const std::unordered_map<std::string, size_t> &facet_to_id, const uint32_t seq_id) const;

    void get_facet_to_index(std::unordered_map<std::string, size_t>& facet_to_index);

    void eq_str_filter_plain(const uint32_t *strt_ids, size_t strt_ids_size,
                             const std::vector<art_leaf *> &query_suggestion,
                             uint32_t *exact_strt_ids, size_t& exact_strt_size) const;

    void scrub_reindex_doc(nlohmann::json& update_doc, nlohmann::json& del_doc, nlohmann::json& old_doc);

    void tokenize_doc_field(const nlohmann::json& document, const field& search_field, std::vector<std::string>& tokens);

    template<typename T>
    bool _arrays_match(std::vector<T> reindex_vals, std::vector<T> old_vals) {
        if(old_vals.size() != reindex_vals.size()) {
            return false;
        }

        for(size_t i=0; i < reindex_vals.size(); i++) {
            const T& reindex_val = reindex_vals[i];
            const T& old_val = old_vals[i];
            if(reindex_val != old_val) {
                return false;
            }
        }

        return true;
    }
};

