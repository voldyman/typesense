#pragma once

#ifndef SSE_NEON
#include <emmintrin.h>

#else
// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L111-L123
#if defined(__GNUC__) || defined(__clang__)
#	pragma push_macro("FORCE_INLINE")
#	pragma push_macro("ALIGN_STRUCT")
#	define FORCE_INLINE       static inline __attribute__((always_inline))
#	define ALIGN_STRUCT(x)    __attribute__((aligned(x)))
#else
#	error "Macro name collisions may happens with unknown compiler"
#	define FORCE_INLINE       static inline
#	define ALIGN_STRUCT(x)    __declspec(align(x))
#endif

#include <stdint.h>
#include "arm_neon.h"
typedef int32x4_t __m128i;

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L223-L224
#define vreinterpretq_m128i_s8(x) \
	vreinterpretq_s32_s8(x)

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L229-L230
#define vreinterpretq_m128i_s32(x) \
	(x)

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L236-L237
#define vreinterpretq_m128i_u8(x) \
	vreinterpretq_s32_u8(x)

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L249-L250
#define vreinterpretq_s8_m128i(x) \
	vreinterpretq_s8_s32(x)

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L262-L263
#define vreinterpretq_u8_m128i(x) \
	vreinterpretq_u8_s32(x)



// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L1325-L1330
// added by hasindu 
// Compares the 16 signed or unsigned 8-bit integers in a and the 16 signed or unsigned 8-bit integers in b for equality. https://msdn.microsoft.com/en-us/library/windows/desktop/bz5xk21a(v=vs.90).aspx
FORCE_INLINE __m128i _mm_cmpeq_epi8 (__m128i a, __m128i b)
{
	return vreinterpretq_m128i_u8(vceqq_s8(vreinterpretq_s8_m128i(a), vreinterpretq_s8_m128i(b)));
}

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L356-L361
// following added by hasindu 
// Sets the 16 signed 8-bit integer values to b.https://msdn.microsoft.com/en-us/library/6e14xhyf(v=vs.100).aspx
FORCE_INLINE __m128i _mm_set1_epi8(char w)
{
	return vreinterpretq_m128i_s8(vdupq_n_s8(w));
}

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L1545-L1550
// added by hasindu (verify this for requirement of alignment)
// Loads 128-bit value. : https://msdn.microsoft.com/zh-cn/library/f4k12ae8(v=vs.90).aspx
FORCE_INLINE __m128i _mm_loadu_si128(const __m128i *p)
{
	return vreinterpretq_m128i_s32(vld1q_s32((int32_t *)p));
}

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L1340-L1345
// added by hasindu 
// Compares the 16 signed 8-bit integers in a and the 16 signed 8-bit integers in b for lesser than. https://msdn.microsoft.com/en-us/library/windows/desktop/9s46csht(v=vs.90).aspx
FORCE_INLINE __m128i _mm_cmplt_epi8 (__m128i a, __m128i b)
{
	return vreinterpretq_m128i_u8(vcltq_s8(vreinterpretq_s8_m128i(a), vreinterpretq_s8_m128i(b)));
}

// source: https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h#L994-L1022
// NEON does not provide a version of this function, here is an article about some ways to repro the results.
// http://stackoverflow.com/questions/11870910/sse-mm-movemask-epi8-equivalent-method-for-arm-neon
// Creates a 16-bit mask from the most significant bits of the 16 signed or unsigned 8-bit integers in a and zero extends the upper bits. https://msdn.microsoft.com/en-us/library/vstudio/s090c8fk(v=vs.100).aspx
FORCE_INLINE int _mm_movemask_epi8(__m128i _a)
{
	uint8x16_t input = vreinterpretq_u8_m128i(_a);
	static const int8_t __attribute__((aligned(16))) xr[8] = { -7, -6, -5, -4, -3, -2, -1, 0 };
	uint8x8_t mask_and = vdup_n_u8(0x80);
	int8x8_t mask_shift = vld1_s8(xr);

	uint8x8_t lo = vget_low_u8(input);
	uint8x8_t hi = vget_high_u8(input);

	lo = vand_u8(lo, mask_and);
	lo = vshl_u8(lo, mask_shift);

	hi = vand_u8(hi, mask_and);
	hi = vshl_u8(hi, mask_shift);

	lo = vpadd_u8(lo, lo);
	lo = vpadd_u8(lo, lo);
	lo = vpadd_u8(lo, lo);

	hi = vpadd_u8(hi, hi);
	hi = vpadd_u8(hi, hi);
	hi = vpadd_u8(hi, hi);

	return ((hi[0] << 8) | (lo[0] & 0xFF));
}

#endif
