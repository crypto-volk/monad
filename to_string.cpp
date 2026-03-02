#include "/home/malvarez/includes/random.hpp"
#include "category/core/runtime/uint256.hpp"
#include <charconv>
#include <immintrin.h>

using monad::vm::runtime::uint256_t;

#undef assert
#if false
    #define assert(x)                                                          \
        do {                                                                   \
            if (!(x)) {                                                        \
                std::cerr << "Assertion failed at line " << __LINE__ << ": "   \
                          << #x << std::endl;                                  \
                std::abort();                                                  \
            }                                                                  \
        }                                                                      \
        while (0)
#else
    #define assert(x)                                                          \
        do {                                                                   \
        }                                                                      \
        while (0)
#endif

template <size_t Base>
[[gnu::noinline]] inline constexpr void
to_string_slow(uint256_t const &x, std::string &buffer)
{
    buffer = x.to_string(Base);
}

constexpr uint64_t pow10e19_1 = 10'000'000'000'000'000'000ul; // 10^19
constexpr std::array<uint64_t, 2> pow10e19_2 = ([]() {
    auto pow2 = monad::vm::runtime::exp(pow10e19_1, 2);
    assert(pow2.as_words()[2] == 0);
    assert(pow2.as_words()[3] == 0);
    return std::array<uint64_t, 2>{pow2.as_words()[0], pow2.as_words()[1]};
})();

constexpr std::array<uint256_t, 5> pow10e19 = ([]() {
    std::array<uint256_t, 5> table{};
    table[0] = uint256_t(1);
    for (size_t i = 1; i < 5; i++) {
        table[i] = table[i - 1] * pow10e19_1;
    }
    return table;
})();

constexpr std::array<char, 16> int_to_pow2_digit = ([]() {
    std::array<char, 16> table{};
    for (size_t i = 0; i < 16; i++) {
        table[i] = static_cast<char>(i < 10 ? '0' + i : 'a' + (i - 10));
    }
    return table;
})();

constexpr auto digits_0_9 = ([]() constexpr {
    std::array<char, 10> digits;
    for (size_t i = 0; i < 10; i++) {
        digits[i] = int_to_pow2_digit[i];
    }
    return digits;
})();

constexpr auto digits_0_99 = ([]() constexpr {
    std::array<char, 200> digits;
    for (size_t i = 0; i < 100; i++) {
        digits[i * 2] = int_to_pow2_digit[i / 10];
        digits[i * 2 + 1] = int_to_pow2_digit[i % 10];
    }
    return digits;
})();

constexpr std::array<uint64_t, 20> pow_10{
    1,
    10,
    100,
    1'000,
    10'000,
    100'000,
    1'000'000,
    10'000'000,
    100'000'000,
    1'000'000'000,
    10'000'000'000,
    100'000'000'000,
    1'000'000'000'000,
    10'000'000'000'000,
    100'000'000'000'000,
    1'000'000'000'000'000,
    10'000'000'000'000'000,
    100'000'000'000'000'000,
    1'000'000'000'000'000'000,
    10'000'000'000'000'000'000ul,
};

constexpr std::array<uint64_t, 19> pow_100{
    1,
    100,
    10'000,
    1'000'000,
    100'000'000,
    10'000'000'000,
    1'000'000'000'000,
    100'000'000'000'000,
    10'000'000'000'000'000,
    1'000'000'000'000'000'000,
};

inline constexpr size_t num_digits_base10_countlz(uint64_t x)
{
    assert(x < pow10e19_1);
    auto const num_bits = static_cast<size_t>(64 - std::countl_zero(x));
    auto num_digits = (num_bits * 1233) >> 12;
    num_digits += (x >= pow_10[num_digits]);
    return num_digits;
}

[[gnu::always_inline]]
static inline __m128i sse_divmod_10000(__m128i x)
{
    constexpr uint32_t div_multiplier = 0xd1b71759;
    constexpr size_t div_shift = 45;
    __m128i quot = _mm_srli_epi64(
        _mm_mul_epu32(x, _mm_set1_epi32(static_cast<int32_t>(div_multiplier))),
        div_shift);
    __m128i rem = _mm_sub_epi32(x, _mm_mul_epu32(quot, _mm_set1_epi32(10000)));
    return _mm_or_si128(quot, _mm_slli_epi64(rem, 32));
}

[[gnu::always_inline]]
static inline __m128i sse_divmod_100(__m128i x)
{
    constexpr uint16_t div_multiplier = 0x147b;
    constexpr size_t div_shift = 3;
    __m128i quot = _mm_srli_epi16(
        _mm_mulhi_epu16(x, _mm_set1_epi16(div_multiplier)), div_shift);
    __m128i rem = _mm_sub_epi16(x, _mm_mullo_epi16(quot, _mm_set1_epi16(100)));
    return _mm_or_si128(quot, _mm_slli_epi32(rem, 16));
}

[[gnu::always_inline]]
static inline __m128i sse_divmod_10(__m128i x)
{
    constexpr uint16_t div_multiplier = 0x199a;
    __m128i quot = _mm_mulhi_epu16(x, _mm_set1_epi16(div_multiplier));
    __m128i rem = _mm_sub_epi16(x, _mm_mullo_epi16(quot, _mm_set1_epi16(10)));
    return _mm_or_si128(quot, _mm_slli_epi16(rem, 8));
}

[[gnu::always_inline]]
static inline __m128i digits16_to_ascii(uint32_t hi, uint32_t lo)
{
    __m128i x = _mm_set_epi64x(lo, hi);

    // 2 base 100'000'000 digits -> 4 base 10'000 digits
    __m128i digits_10000 = sse_divmod_10000(x);

    // 4 base 10'000 digits -> 8 base 100 digits
    __m128i digits_100 = sse_divmod_100(digits_10000);

    // 8 base 100 digits -> 16 base 10 digits
    __m128i digits_10 = sse_divmod_10(digits_100);

    return _mm_add_epi8(digits_10, _mm_set1_epi8('0'));
}

template <bool print_leading_zeros>
inline void write_base10_sse(uint64_t x, std::string &buffer)
{
    size_t digits_base10;
    if constexpr (print_leading_zeros) {
        digits_base10 = 19;
    }
    else {
        digits_base10 = num_digits_base10_countlz(x);
    }
    size_t I = buffer.length();
    buffer.resize(I + digits_base10);

    if constexpr (!print_leading_zeros) {
        if (x < 1000) {
            // Number takes at most 3 digits; print those and return.
            if (x >= 10) {
                // At least two-digit tail
                if (x >= 100) {
                    // Three-digit tail
                    buffer[I + digits_base10 - 3] = digits_0_9[x / 100];
                    x = x % 100;
                }
                std::memcpy(
                    &buffer[I + digits_base10 - 2], &digits_0_99[x * 2], 2);
            }
            else if (x) {
                // One-digit tail
                buffer[I + digits_base10 - 1] = digits_0_9[x];
            }
            return;
        }
    }
    uint64_t head = x / 1000;
    uint64_t tail = x % 1000;

    // Print tail unconditionally
    buffer[I + digits_base10 - 3] = digits_0_9[tail / 100];
    tail = tail % 100;
    std::memcpy(&buffer[I + digits_base10 - 2], &digits_0_99[tail * 2], 2);

    auto head_hi = uint32_t(head / 100'000'000);
    auto head_lo = uint32_t(head % 100'000'000);
    __m128i head_digits = digits16_to_ascii(head_hi, head_lo);

    if constexpr (print_leading_zeros) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&buffer[I]), head_digits);
    }
    else {
        alignas(16) char scratch[16];
        _mm_store_si128(reinterpret_cast<__m128i *>(scratch), head_digits);
        std::memcpy(
            &buffer[I], &scratch[16 - (digits_base10 - 3)], digits_base10 - 3);
    }
}

template <bool print_leading_zeros>
struct WriteBase10SSE
{
    [[gnu::always_inline]]
    static inline void write_digits(uint64_t x, std::string &buffer)
    {
        write_base10_sse<print_leading_zeros>(x, buffer);
    }
};

// ============================================================================
// AVX2 versions of the SSE helpers (same logic, 256-bit registers)
// ============================================================================

[[gnu::always_inline]]
static inline __m256i avx2_divmod_10000(__m256i x)
{
    constexpr uint32_t div_multiplier = 0xd1b71759;
    constexpr size_t div_shift = 45;
    __m256i quot = _mm256_srli_epi64(
        _mm256_mul_epu32(
            x, _mm256_set1_epi32(static_cast<int32_t>(div_multiplier))),
        div_shift);
    __m256i rem =
        _mm256_sub_epi32(x, _mm256_mul_epu32(quot, _mm256_set1_epi32(10000)));
    return _mm256_or_si256(quot, _mm256_slli_epi64(rem, 32));
}

[[gnu::always_inline]]
static inline __m256i avx2_divmod_100(__m256i x)
{
    constexpr uint16_t div_multiplier = 0x147b;
    constexpr size_t div_shift = 3;
    __m256i quot = _mm256_srli_epi16(
        _mm256_mulhi_epu16(x, _mm256_set1_epi16(div_multiplier)), div_shift);
    __m256i rem =
        _mm256_sub_epi16(x, _mm256_mullo_epi16(quot, _mm256_set1_epi16(100)));
    return _mm256_or_si256(quot, _mm256_slli_epi32(rem, 16));
}

[[gnu::always_inline]]
static inline __m256i avx2_divmod_10(__m256i x)
{
    constexpr uint16_t div_multiplier = 0x199a;
    __m256i quot = _mm256_mulhi_epu16(x, _mm256_set1_epi16(div_multiplier));
    __m256i rem =
        _mm256_sub_epi16(x, _mm256_mullo_epi16(quot, _mm256_set1_epi16(10)));
    return _mm256_or_si256(quot, _mm256_slli_epi16(rem, 8));
}

[[gnu::always_inline]]
static inline __m128i digits16_to_ascii_avx2(uint32_t hi, uint32_t lo)
{
    __m256i x = _mm256_set_epi64x(0, 0, lo, hi);

    // 2 base 100'000'000 digits -> 4 base 10'000 digits
    __m256i digits_10000 = avx2_divmod_10000(x);

    // 4 base 10'000 digits -> 8 base 100 digits
    __m256i digits_100 = avx2_divmod_100(digits_10000);

    // 8 base 100 digits -> 16 base 10 digits
    __m256i digits_10 = avx2_divmod_10(digits_100);

    __m256i ascii = _mm256_add_epi8(digits_10, _mm256_set1_epi8('0'));
    return _mm256_castsi256_si128(ascii);
}

template <bool print_leading_zeros>
inline void write_base10_avx2(uint64_t x, std::string &buffer)
{
    size_t digits_base10;
    if constexpr (print_leading_zeros) {
        digits_base10 = 19;
    }
    else {
        digits_base10 = num_digits_base10_countlz(x);
    }
    size_t I = buffer.length();
    buffer.resize(I + digits_base10);

    if constexpr (!print_leading_zeros) {
        if (x < 1000) {
            if (x >= 10) {
                if (x >= 100) {
                    buffer[I + digits_base10 - 3] = digits_0_9[x / 100];
                    x = x % 100;
                }
                std::memcpy(
                    &buffer[I + digits_base10 - 2], &digits_0_99[x * 2], 2);
            }
            else if (x) {
                buffer[I + digits_base10 - 1] = digits_0_9[x];
            }
            return;
        }
    }
    uint64_t head = x / 1000;
    uint64_t tail = x % 1000;

    // Print tail unconditionally
    buffer[I + digits_base10 - 3] = digits_0_9[tail / 100];
    tail = tail % 100;
    std::memcpy(&buffer[I + digits_base10 - 2], &digits_0_99[tail * 2], 2);

    auto head_hi = uint32_t(head / 100'000'000);
    auto head_lo = uint32_t(head % 100'000'000);
    __m128i head_digits = digits16_to_ascii_avx2(head_hi, head_lo);

    if constexpr (print_leading_zeros) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&buffer[I]), head_digits);
    }
    else {
        alignas(16) char scratch[16];
        _mm_store_si128(reinterpret_cast<__m128i *>(scratch), head_digits);
        std::memcpy(
            &buffer[I], &scratch[16 - (digits_base10 - 3)], digits_base10 - 3);
    }
}

template <bool print_leading_zeros>
struct WriteBase10AVX2
{
    [[gnu::always_inline]]
    static inline void write_digits(uint64_t x, std::string &buffer)
    {
        write_base10_avx2<print_leading_zeros>(x, buffer);
    }
};

// Process two 16-digit heads in parallel, returning 32 ASCII digits
[[gnu::always_inline]]
static inline __m256i
digits32_to_ascii(uint32_t hi_a, uint32_t lo_a, uint32_t hi_b, uint32_t lo_b)
{
    // Lower 128 bits: limb A (hi_a, lo_a) → 16 ASCII digits
    // Upper 128 bits: limb B (hi_b, lo_b) → 16 ASCII digits
    __m256i x = _mm256_set_epi64x(lo_b, hi_b, lo_a, hi_a);
    __m256i digits_10000 = avx2_divmod_10000(x);
    __m256i digits_100 = avx2_divmod_100(digits_10000);
    __m256i digits_10 = avx2_divmod_10(digits_100);
    return _mm256_add_epi8(digits_10, _mm256_set1_epi8('0'));
}

// Write two leading-zero limbs (19 digits each) using AVX2
inline void write_base10_avx2_pair(uint64_t a, uint64_t b, std::string &buffer)
{
    size_t I = buffer.length();
    buffer.resize(I + 38);

    // Split both limbs into head (16 digits) + tail (3 digits)
    uint64_t head_a = a / 1000;
    uint64_t tail_a = a % 1000;
    uint64_t head_b = b / 1000;
    uint64_t tail_b = b % 1000;

    // Scalar tails for limb A
    buffer[I + 16] = digits_0_9[tail_a / 100];
    std::memcpy(&buffer[I + 17], &digits_0_99[(tail_a % 100) * 2], 2);

    // Scalar tails for limb B
    buffer[I + 19 + 16] = digits_0_9[tail_b / 100];
    std::memcpy(&buffer[I + 19 + 17], &digits_0_99[(tail_b % 100) * 2], 2);

    // AVX2: convert both 16-digit heads to ASCII in parallel
    auto head_a_hi = uint32_t(head_a / 100'000'000);
    auto head_a_lo = uint32_t(head_a % 100'000'000);
    auto head_b_hi = uint32_t(head_b / 100'000'000);
    auto head_b_lo = uint32_t(head_b % 100'000'000);

    __m256i digits =
        digits32_to_ascii(head_a_hi, head_a_lo, head_b_hi, head_b_lo);

    // Store lower 128 bits → limb A head, upper 128 bits → limb B head
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(&buffer[I]),
        _mm256_castsi256_si128(digits));
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(&buffer[I + 19]),
        _mm256_extracti128_si256(digits, 1));
}

template <bool write_leading_zeros>
inline constexpr void
write_to_string_base10_lr_switch(uint64_t x, std::string &buffer)
{
    // assert(x < max_pow10_in_uint64_t);
    size_t digits_base10;
    if constexpr (write_leading_zeros) {
        digits_base10 = 19;
    }
    else {
        digits_base10 = num_digits_base10_countlz(x);
    }
    size_t I = buffer.length();
    buffer.resize(I + digits_base10);
    size_t i = 0;
    size_t digits_base100 = digits_base10 / 2;
    bool extra_digit = digits_base10 % 2;
    if (extra_digit) {
        uint64_t msd = x / pow_10[digits_base10 - 1];
        x -= msd * pow_10[digits_base10 - 1];
        buffer[I + i] = int_to_pow2_digit[msd];
        i++;
    }

#define CASE(N)                                                                \
    {                                                                          \
        uint64_t msd = x / pow_100[(N) - 1];                                   \
        x -= msd * pow_100[(N) - 1];                                           \
        std::memcpy(&buffer[I + i], &digits_0_99[msd * 2], 2);                 \
        i += 2;                                                                \
    }                                                                          \
    [[fallthrough]];
    switch (digits_base100) {
    case 10:
        CASE(10)
    case 9:
        CASE(9)
    case 8:
        CASE(8)
    case 7:
        CASE(7)
    case 6:
        CASE(6)
    case 5:
        CASE(5)
    case 4:
        CASE(4)
    case 3:
        CASE(3)
    case 2:
        CASE(2)
    case 1:
        CASE(1)
    default:
        break;
    }
}

template <bool print_leading_zeros>
struct WriteBase10LRSwitch
{
    [[gnu::always_inline]]
    static inline constexpr void write_digits(uint64_t x, std::string &buffer)
    {
        write_to_string_base10_lr_switch<print_leading_zeros>(x, buffer);
    }
};

template <bool print_leading_zeros, size_t digits>
inline void
write_to_string_base10_jeaii_32(std::uint32_t n, std::string &buffer)
{
    constexpr auto mask = (std::uint64_t(1) << 57) - 1;
    auto y = n * std::uint64_t(1'441'151'881);
    auto const I = buffer.length();

    size_t num_digits;
    if constexpr (print_leading_zeros) {
        num_digits = digits;
    }
    else {
        num_digits = num_digits_base10_countlz(n);
    }
    buffer.resize(I + num_digits);

    size_t num_digits_base100 = num_digits / 2;
    size_t extra_digit = num_digits % 2;
    size_t skip_digits_base100 = 5 - num_digits_base100 - extra_digit;
    for (size_t i = 0; i < skip_digits_base100; i++) {
        y &= mask;
        y *= 100;
    }
    if (extra_digit) {
        buffer[I + 0] = digits_0_99[size_t(y >> 57) * 2];
        y &= mask;
        y *= 100;
    }
    if constexpr (print_leading_zeros) {
#pragma GCC unroll 5
        for (size_t i = 0; i < num_digits_base100; i++) {
            std::memcpy(
                &buffer[I + 2 * i + extra_digit],
                &digits_0_99[size_t(y >> 57) * 2],
                2);
            y &= mask;
            y *= 100;
        }
    }
    else {
        for (size_t i = 0; i < num_digits_base100; i++) {
            std::memcpy(
                &buffer[I + 2 * i + extra_digit],
                &digits_0_99[size_t(y >> 57) * 2],
                2);
            y &= mask;
            y *= 100;
        }
    }
}

template <bool print_leading_zeros>
inline constexpr void
write_to_string_base10_jeaiii(uint64_t x, std::string &buffer)
{
    constexpr uint64_t max = 1'000'000'000ul;

    uint32_t high = static_cast<uint32_t>(x / max);
    uint32_t low = static_cast<uint32_t>(x % max);
    write_to_string_base10_jeaii_32<print_leading_zeros, 9>(high, buffer);
    if constexpr (print_leading_zeros) {
        write_to_string_base10_jeaii_32<true, 10>(low, buffer);
    }
    else if (high != 0) {
        write_to_string_base10_jeaii_32<true, 10>(low, buffer);
    }
    else {
        write_to_string_base10_jeaii_32<print_leading_zeros, 10>(low, buffer);
    }
}

template <bool print_leading_zeros>
struct WriteBase10Jeaiii
{
    [[gnu::always_inline]]
    static inline constexpr void write_digits(uint64_t x, std::string &buffer)
    {
        write_to_string_base10_jeaiii<print_leading_zeros>(x, buffer);
    }
};

// magick numbers for 16-bit division
#define DIV_10 0x199a // shift = 0 + 16
#define DIV_100 0x147b // shift = 3 + 16

// magic number for 32-bit division
#define DIV_10000 0xd1b71759 // shift = 13 + 32

template <bool print_leading_zeros>
inline constexpr void
write_to_string_base10_sse(uint64_t v, std::string &buffer)
{
    size_t digits_base10;
    if constexpr (print_leading_zeros) {
        digits_base10 = 19;
    }
    else {
        digits_base10 = num_digits_base10_countlz(v);
    }
    size_t I = buffer.length();
    buffer.resize(I + digits_base10);
    // v is 16-digit number = abcdefghijklmnop
    __m128i const div_10000 = _mm_set1_epi32(DIV_10000);
    __m128i const mul_10000 = _mm_set1_epi32(10000);
    int const div_10000_shift = 45;

    __m128i const div_100 = _mm_set1_epi16(DIV_100);
    __m128i const mul_100 = _mm_set1_epi16(100);
    int const div_100_shift = 3;

    __m128i const div_10 = _mm_set1_epi16(DIV_10);
    __m128i const mul_10 = _mm_set1_epi16(10);

    __m128i const ascii0 = _mm_set1_epi8('0');

    // can't be easliy done in SSE
    uint32_t const a = v / 100000000; // 8-digit number: abcdefgh
    uint32_t const b = v % 100000000; // 8-digit number: ijklmnop

    //                [ 3 | 2 | 1 | 0 | 3 | 2 | 1 | 0 | 3 | 2 | 1 | 0 | 3 | 2 |
    //                1 | 0 ]
    // x            = [       0       |      ijklmnop |       0       | abcdefgh
    // ]
    __m128i x = _mm_set_epi64x(b, a);

    // x div 10^4   = [       0       |          ijkl |       0       | abcd ]
    __m128i x_div_10000;
    x_div_10000 = _mm_mul_epu32(x, div_10000);
    x_div_10000 = _mm_srli_epi64(x_div_10000, div_10000_shift);

    // x mod 10^4   = [       0       |          mnop |       0       | efgh ]
    __m128i x_mod_10000;
    x_mod_10000 = _mm_mul_epu32(x_div_10000, mul_10000);
    x_mod_10000 = _mm_sub_epi32(x, x_mod_10000);

    // y            = [          mnop |          ijkl |          efgh | abcd ]
    __m128i y = _mm_or_si128(x_div_10000, _mm_slli_epi64(x_mod_10000, 32));

    // y_div_100    = [   0   |    mn |   0   |    ij |   0   |    ef |   0   |
    // ab ]
    __m128i y_div_100;
    y_div_100 = _mm_mulhi_epu16(y, div_100);
    y_div_100 = _mm_srli_epi16(y_div_100, div_100_shift);

    // y_mod_100    = [   0   |    op |   0   |    kl |   0   |    gh |   0   |
    // cd ]
    __m128i y_mod_100;
    y_mod_100 = _mm_mullo_epi16(y_div_100, mul_100);
    y_mod_100 = _mm_sub_epi16(y, y_mod_100);

    // z            = [    mn |    op |    ij |    kl |    ef |    gh |    ab |
    // cd ]
    __m128i z = _mm_or_si128(y_div_100, _mm_slli_epi32(y_mod_100, 16));

    // z_div_10     = [ 0 | m | 0 | o | 0 | i | 0 | k | 0 | e | 0 | g | 0 | a |
    // 0 | c ]
    __m128i z_div_10 = _mm_mulhi_epu16(z, div_10);

    // z_mod_10     = [ 0 | n | 0 | p | 0 | j | 0 | l | 0 | f | 0 | h | 0 | b |
    // 0 | d ]
    __m128i z_mod_10;
    z_mod_10 = _mm_mullo_epi16(z_div_10, mul_10);
    z_mod_10 = _mm_sub_epi16(z, z_mod_10);

    // interleave z_mod_10 and z_div_10 -
    // tmp          = [ m | n | o | p | i | j | k | l | e | f | g | h | a | b |
    // c | d ]
    __m128i tmp = _mm_or_si128(z_div_10, _mm_slli_epi16(z_mod_10, 8));

    // determine number of leading zeros
    uint16_t mask =
        ~_mm_movemask_epi8(_mm_cmpeq_epi8(tmp, _mm_setzero_si128()));
    uint32_t offset = __builtin_ctz(mask | 0x8000);

    // convert to ascii
    tmp = _mm_add_epi8(tmp, ascii0);

    // and save result
    _mm_storeu_si128((__m128i *)&buffer[I], tmp);
}

template <bool print_leading_zeros>
struct WriteBase10SSE16
{
    [[gnu::always_inline]]
    static inline constexpr void write_digits(uint64_t x, std::string &buffer)
    {
        write_to_string_base10_sse<print_leading_zeros>(x, buffer);
    }
};

template <bool print_leading_zeros>
inline constexpr void write_to_string_base10_rl(uint64_t x, std::string &buffer)
{
    size_t digits_base10;
    if constexpr (print_leading_zeros) {
        digits_base10 = 19;
    }
    else {
        digits_base10 = num_digits_base10_countlz(x);
    }
    size_t I = buffer.length();
    buffer.resize(I + digits_base10);

    char *ptr = buffer.data() + I + digits_base10 - 2;
    while (x >= 100) {
        uint64_t lsd = x % 100;
        x /= 100;
        std::memcpy(ptr, &digits_0_99[lsd * 2], 2);
        ptr -= 2;
    }
    bool extra_digit = digits_base10 % 2;
    if (extra_digit) {
        ptr[1] = int_to_pow2_digit[x % 10];
        x /= 10;
    }
}

template <bool print_leading_zeros>
struct WriteBase10RL
{
    [[gnu::always_inline]]
    static inline constexpr void write_digits(uint64_t x, std::string &buffer)
    {
        write_to_string_base10_rl<print_leading_zeros>(x, buffer);
    }
};

constexpr std::array<uint64_t, 20> pow10 = []() {
    std::array<uint64_t, 20> table{};
    table[0] = 1;
    for (size_t i = 1; i < 20; i++) {
        table[i] = table[i - 1] * 10;
    }
    return table;
}();

template <bool print_leading_zeros>
inline constexpr void
write_to_string_base10_alvarez(uint64_t x, std::string &buffer)
{
    // x has at most 19 digits
    assert(x < pow10e19_1);
    size_t I = buffer.length();
    if constexpr (print_leading_zeros) {
        buffer.resize(I + 19);
        uint64_t w18_16 = x / 10'000'000'000'000'000;
        uint64_t w15_0 = x % 10'000'000'000'000'000;

        uint64_t w_hi = w15_0 / 100'000'000;
        uint64_t w_lo = x % 100'000'000;

        uint64_t w18 = w18_16 / 100;
        uint64_t w17_16 = w18_16 % 100;
        std::memcpy(&buffer[I + 0], &digits_0_99[w18 * 2 + 1], 1);
        std::memcpy(&buffer[I + 1], &digits_0_99[w17_16 * 2], 2);
        size_t index_hi = I + 6 + 3;
        size_t index_lo = index_hi + 8;
#pragma GCC unroll 8
        for (size_t i = 0; i < 4; i++) {
            uint64_t digit_hi = w_hi % 100;
            w_hi /= 100;
            uint64_t digit_lo = w_lo % 100;
            w_lo /= 100;
            std::memcpy(
                &buffer[index_hi - 2 * i], &digits_0_99[digit_hi * 2], 2);
            std::memcpy(
                &buffer[index_lo - 2 * i], &digits_0_99[digit_lo * 2], 2);
        }
    }
    else {
        /*
        size_t digits_base10 = num_digits_base10_countlz(x);
        buffer.resize(I + digits_base10);
        // Single loop until a power of two does fewer multiplications but has
        less parallelism if (digits_base10 % 2) { buffer[I + digits_base10 - 1]
        = int_to_pow2_digit[x % 10]; x /= 10;
        }
        size_t half_digits = digits_base10 / 2;
        uint64_t hi = x / pow_100[half_digits];
        uint64_t lo = x % pow_100[half_digits];
        for (size_t i = 0; i < half_digits / 2; i++) {
        }
        */
        write_to_string_base10_rl<false>(x, buffer);
    }
}

template <bool print_leading_zeros>
struct WriteBase10Alvarez
{
    [[gnu::always_inline]]
    static inline constexpr void write_digits(uint64_t x, std::string &buffer)
    {
        write_to_string_base10_alvarez<print_leading_zeros>(x, buffer);
    }
};

template <bool print_leading_zeros>
inline constexpr void
write_to_string_base10_lemire(uint64_t x, std::string &buffer)
{
    // x has at most 19 digits
    assert(x < pow10e19_1);
    size_t I = buffer.length();
    if constexpr (print_leading_zeros) {
        buffer.resize(I + 19);
        uint64_t w18_8 = x / 100'000'000;
        uint64_t w7_0 = x % 100'000'000;
        uint64_t w18_12 = w18_8 / 10'000;
        uint64_t w11_8 = w18_8 % 10'000;
        uint64_t w7_4 = w7_0 / 10'000;
        uint64_t w3_0 = w7_0 % 10'000;
        uint64_t w18_14 = w18_12 / 100;
        uint64_t w18_18 = w18_14 / 10'000;
        uint64_t w17_14 = w18_14 % 10'000;
        uint64_t w17_16 = w17_14 / 100;
        uint64_t w15_14 = w17_14 % 100;
        uint64_t w13_12 = w18_12 % 100;
        uint64_t w11_10 = w11_8 / 100;
        uint64_t w9_8 = w11_8 % 100;
        uint64_t w7_6 = w7_4 / 100;
        uint64_t w5_4 = w7_4 % 100;
        uint64_t w3_2 = w3_0 / 100;
        uint64_t w1_0 = w3_0 % 100;
        std::memcpy(&buffer[I + 0], &digits_0_99[w18_18 * 2 + 1], 1);
        std::memcpy(&buffer[I + 1], &digits_0_99[w17_16 * 2], 2);
        std::memcpy(&buffer[I + 3], &digits_0_99[w15_14 * 2], 2);
        std::memcpy(&buffer[I + 5], &digits_0_99[w13_12 * 2], 2);
        std::memcpy(&buffer[I + 7], &digits_0_99[w11_10 * 2], 2);
        std::memcpy(&buffer[I + 9], &digits_0_99[w9_8 * 2], 2);
        std::memcpy(&buffer[I + 11], &digits_0_99[w7_6 * 2], 2);
        std::memcpy(&buffer[I + 13], &digits_0_99[w5_4 * 2], 2);
        std::memcpy(&buffer[I + 15], &digits_0_99[w3_2 * 2], 2);
        std::memcpy(&buffer[I + 17], &digits_0_99[w1_0 * 2], 2);
    }
    else {
        /*
        size_t digits_base10 = num_digits_base10_countlz(x);
        buffer.resize(I + digits_base10);
        */
        write_to_string_base10_rl<false>(x, buffer);
    }
}

template <bool print_leading_zeros>
struct WriteBase10Lemire
{
    [[gnu::always_inline]]
    static inline constexpr void write_digits(uint64_t x, std::string &buffer)
    {
        write_to_string_base10_lemire<print_leading_zeros>(x, buffer);
    }
};

template <template <bool print_leading_zeros> typename Writer>
[[gnu::noinline]]
inline constexpr void
to_string_base10_variable_faster(uint256_t const &x, std::string &buffer)
{
    auto const num_bits = bit_width(x);
    if (num_bits == 0) {
        buffer = "0";
        return;
    }
    buffer.reserve(1 + (num_bits * 1233 >> 12)); // log10(2) ~= 1233/4096

    if (x < pow10e19[3]) {
        assert(!x[3]);
        if (x < pow10e19[1]) {
            Writer<false>::write_digits(x[0], buffer);
        }
        else {
            if (x < pow10e19[2]) {
                assert(!x[2]);
                auto [w1, w0] = monad::vm::runtime::div(x[1], x[0], pow10e19_1);
                Writer<false>::write_digits(w1, buffer);
                Writer<true>::write_digits(w0, buffer);
            }
            else {
                assert(!x[3]);
                std::array<uint64_t, 3> x_words{x[0], x[1], x[2]};
                auto [w21, w0] =
                    monad::vm::runtime::long_div<3>(x_words, pow10e19_1);
                assert(!w21[2]);
                auto [w2, w1] =
                    monad::vm::runtime::div(w21[1], w21[0], pow10e19_1);
                Writer<false>::write_digits(w2, buffer);
                Writer<true>::write_digits(w1, buffer);
                Writer<true>::write_digits(w0, buffer);
            }
        }
    }
    else {
        std::array<uint64_t, 4> const &x_words = x.as_words();
        if (x < pow10e19[4]) {
            auto [hi, lo] = monad::vm::runtime::udivrem(x_words, pow10e19_2);
            assert(!hi[2]);

            auto [w3, w2] = monad::vm::runtime::div(hi[1], hi[0], pow10e19_1);
            Writer<false>::write_digits(w3, buffer);
            Writer<true>::write_digits(w2, buffer);

            auto [w1, w0] = monad::vm::runtime::div(lo[1], lo[0], pow10e19_1);
            Writer<true>::write_digits(w1, buffer);
            Writer<true>::write_digits(w0, buffer);
        }
        else {
            auto [hi_, lo] = monad::vm::runtime::udivrem(x_words, pow10e19_2);
            assert(!hi_[3]);

            auto hi = std::array<uint64_t, 3>{hi_[0], hi_[1], hi_[2]};
            auto [w43, w2] = monad::vm::runtime::long_div(hi, pow10e19_1);
            assert(!w43[2]);
            auto [w4, w3] = monad::vm::runtime::div(w43[1], w43[0], pow10e19_1);
            Writer<false>::write_digits(w4, buffer);
            Writer<true>::write_digits(w3, buffer);
            Writer<true>::write_digits(w2, buffer);

            auto [w1, w0] = monad::vm::runtime::div(lo[1], lo[0], pow10e19_1);
            Writer<true>::write_digits(w1, buffer);
            Writer<true>::write_digits(w0, buffer);
        }
    }
}

// Tag type for the AVX2 batched specialization; single-limb falls back to SSE
template <bool print_leading_zeros>
struct WriteBase10AVX2Batched
{
    [[gnu::always_inline]]
    static inline void write_digits(uint64_t x, std::string &buffer)
    {
        write_base10_sse<print_leading_zeros>(x, buffer);
    }
};

template <>
[[gnu::noinline]]
inline constexpr void to_string_base10_variable_faster<WriteBase10AVX2Batched>(
    uint256_t const &x, std::string &buffer)
{
    auto const num_bits = bit_width(x);
    if (num_bits == 0) {
        buffer = "0";
        return;
    }
    buffer.reserve(1 + (num_bits * 1233 >> 12));

    if (x < pow10e19[3]) {
        assert(!x[3]);
        if (x < pow10e19[1]) {
            write_base10_sse<false>(x[0], buffer);
        }
        else if (x < pow10e19[2]) {
            assert(!x[2]);
            auto [w1, w0] = monad::vm::runtime::div(x[1], x[0], pow10e19_1);
            write_base10_sse<false>(w1, buffer);
            write_base10_sse<true>(w0, buffer);
        }
        else {
            // 3 limbs: 2 lz limbs → batch (w1, w0)
            assert(!x[3]);
            std::array<uint64_t, 3> x_words{x[0], x[1], x[2]};
            auto [w21, w0] =
                monad::vm::runtime::long_div<3>(x_words, pow10e19_1);
            assert(!w21[2]);
            auto [w2, w1] = monad::vm::runtime::div(w21[1], w21[0], pow10e19_1);
            write_base10_sse<false>(w2, buffer);
            write_base10_avx2_pair(w1, w0, buffer);
        }
    }
    else {
        std::array<uint64_t, 4> const &x_words = x.as_words();
        if (x < pow10e19[4]) {
            // 4 limbs: 3 lz limbs → batch (w1, w0), single w2
            auto [hi, lo] = monad::vm::runtime::udivrem(x_words, pow10e19_2);
            assert(!hi[2]);
            auto [w3, w2] = monad::vm::runtime::div(hi[1], hi[0], pow10e19_1);
            auto [w1, w0] = monad::vm::runtime::div(lo[1], lo[0], pow10e19_1);
            write_base10_sse<false>(w3, buffer);
            write_base10_sse<true>(w2, buffer);
            write_base10_avx2_pair(w1, w0, buffer);
        }
        else {
            // 5 limbs: 4 lz limbs → batch (w3, w2) and (w1, w0)
            auto [hi_, lo] = monad::vm::runtime::udivrem(x_words, pow10e19_2);
            assert(!hi_[3]);
            auto hi = std::array<uint64_t, 3>{hi_[0], hi_[1], hi_[2]};
            auto [w43, w2] = monad::vm::runtime::long_div(hi, pow10e19_1);
            assert(!w43[2]);
            auto [w4, w3] = monad::vm::runtime::div(w43[1], w43[0], pow10e19_1);
            auto [w1, w0] = monad::vm::runtime::div(lo[1], lo[0], pow10e19_1);
            write_base10_sse<false>(w4, buffer);
            write_base10_avx2_pair(w3, w2, buffer);
            write_base10_avx2_pair(w1, w0, buffer);
        }
    }
}

constexpr std::array<uint64_t, 5> divisors{
    // 0-digit tail - these do not matter.
    1,
    // 4-digit tail
    100,
    // 8-digit tail
    10000,
    // 12-digit tail
    1000000,
    // 16-digit tail
    100000000,
};
constexpr std::array<uint8_t, 20> shifts{};

template <bool print_leading_zeros>
inline constexpr void write_base10_picallo(uint64_t x, std::string &buffer)
{
    if constexpr (print_leading_zeros) {
        write_base10_avx2<true>(x, buffer);
    }
    else {
        size_t digits = num_digits_base10_countlz(x);
        size_t I = buffer.length();
        buffer.resize(I + digits);

        size_t index = I + digits - 1;
        while (digits & 3) {
            auto lsd = x % 10;
            x = x / 10;
            buffer[index] = static_cast<char>('0' + lsd);
            index -= 1;
            digits -= 1;
        }
        // Go back one since we're writing 2 byte chunks at a time
        index -= 1;
        size_t digit_pairs = digits >> 1;
        size_t iterations = digits >> 2;
        uint64_t x_hi = x / divisors[iterations];
        uint64_t x_lo = x - x_hi * divisors[iterations];
        for (size_t i = 0; i < iterations; i++) {
            size_t const index_lo = index;
            size_t const index_hi = index - digit_pairs;
            auto const lsb_lo = x_lo % 100;
            auto const lsb_hi = x_hi % 100;
            x_lo /= 100;
            x_hi /= 100;
            std::memcpy(&buffer[index_lo], &digits_0_99[lsb_lo * 2], 2);
            std::memcpy(&buffer[index_hi], &digits_0_99[lsb_hi * 2], 2);
            index -= 2;
        }
    }
}

template <bool print_leading_zeros>
struct WriteBase10Picallo
{
    [[gnu::always_inline]]
    static inline constexpr void write_digits(uint64_t x, std::string &buffer)
    {
        write_base10_picallo<print_leading_zeros>(x, buffer);
    }
};

// ============================================================================
// SSE variant: unconditional SIMD for lower 16 digits, scalar for upper 0-3
// lo_lo derived from x directly to shorten dependency chain
// ============================================================================

template <bool print_leading_zeros>
inline void write_base10_sse_v2(uint64_t x, std::string &buffer)
{
    size_t digits_base10;
    if constexpr (print_leading_zeros) {
        digits_base10 = 19;
    }
    else {
        digits_base10 = num_digits_base10_countlz(x);
    }
    size_t I = buffer.length();
    buffer.resize(I + digits_base10);

    // Depth 1 (two independent results from x):
    uint64_t x_div_1e8 = x / 100'000'000;
    auto lo_lo = uint32_t(x % 100'000'000); // from x, not from lo!

    // Depth 2 (from x_div_1e8):
    uint64_t hi3 = x_div_1e8 / 100'000'000; // 0-999
    auto lo_hi = uint32_t(x_div_1e8 % 100'000'000);

    // SIMD: unconditionally convert lower 16 digits
    __m128i lo_digits = digits16_to_ascii(lo_hi, lo_lo);

    if constexpr (print_leading_zeros) {
        // Always 19 digits: 3 scalar + 16 SIMD
        buffer[I + 0] = digits_0_9[hi3 / 100];
        std::memcpy(&buffer[I + 1], &digits_0_99[(hi3 % 100) * 2], 2);
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(&buffer[I + 3]), lo_digits);
    }
    else {
        if (digits_base10 <= 16) {
            alignas(16) char scratch[16];
            _mm_store_si128(reinterpret_cast<__m128i *>(scratch), lo_digits);
            std::memcpy(
                &buffer[I], &scratch[16 - digits_base10], digits_base10);
        }
        else {
            // 17-19 digits: SIMD for lower 16, scalar for leading 1-3
            _mm_storeu_si128(
                reinterpret_cast<__m128i *>(&buffer[I + digits_base10 - 16]),
                lo_digits);
            size_t extra = digits_base10 - 16;
            if (extra == 3) {
                buffer[I] = digits_0_9[hi3 / 100];
                std::memcpy(&buffer[I + 1], &digits_0_99[(hi3 % 100) * 2], 2);
            }
            else if (extra == 2) {
                std::memcpy(&buffer[I], &digits_0_99[hi3 * 2], 2);
            }
            else {
                buffer[I] = digits_0_9[hi3];
            }
        }
    }
}

template <bool print_leading_zeros>
struct WriteBase10SSEv2
{
    [[gnu::always_inline]]
    static inline void write_digits(uint64_t x, std::string &buffer)
    {
        write_base10_sse_v2<print_leading_zeros>(x, buffer);
    }
};

template <bool print_leading_zeros>
inline void write_base10_to_chars(uint64_t x, std::string &buffer)
{
    if constexpr (print_leading_zeros) {
        size_t I = buffer.length();
        buffer.resize(I + 19);
        // Right-justify into 19 chars, zero-pad the front
        char *begin = &buffer[I];
        char *end = begin + 19;
        auto [ptr, ec] = std::to_chars(begin, end, x);
        // to_chars writes left-justified; shift right and zero-fill
        size_t written = static_cast<size_t>(ptr - begin);
        size_t pad = 19 - written;
        if (pad) {
            std::memmove(begin + pad, begin, written);
            std::memset(begin, '0', pad);
        }
    }
    else {
        size_t I = buffer.length();
        // to_chars needs a destination; write into buffer directly
        // Max 19 digits for a uint64 < 10^19
        buffer.resize(I + num_digits_base10_countlz(x));
        char *begin = &buffer[I];
        char *end = begin + 19;
        auto [ptr, ec] = std::to_chars(begin, end, x);
        // Shrink to actual length
        //buffer.resize(I + static_cast<size_t>(ptr - begin));
    }
}

template <bool print_leading_zeros>
struct WriteBase10ToChars
{
    [[gnu::always_inline]]
    static inline void write_digits(uint64_t x, std::string &buffer)
    {
        write_base10_to_chars<print_leading_zeros>(x, buffer);
    }
};

// Hybrid: to_chars for the first (non-lz) limb, SSE for lz limbs
template <bool print_leading_zeros>
struct WriteBase10Hybrid
{
    [[gnu::always_inline]]
    static inline void write_digits(uint64_t x, std::string &buffer)
    {
        if constexpr (print_leading_zeros) {
            write_base10_sse<true>(x, buffer);
        }
        else {
            write_base10_to_chars<false>(x, buffer);
        }
    }
};

// ============================================================================
// Registry
// ============================================================================

using to_string_fn_t = void (*)(uint256_t const &, std::string &);

struct Implementation
{
    std::string_view name;
    to_string_fn_t fn;
};

#define IMPL(display_name, writer)                                             \
    {display_name, to_string_base10_variable_faster<writer>}
static constexpr Implementation implementations[] = {
    {"slow", to_string_slow<10>},
    IMPL("sse", WriteBase10SSE),
    IMPL("lr_switch", WriteBase10LRSwitch),
    IMPL("jeaiii", WriteBase10Jeaiii),
    IMPL("rl", WriteBase10RL),
    IMPL("alvarez", WriteBase10Alvarez),
    IMPL("lemire", WriteBase10Lemire),
    IMPL("avx2", WriteBase10AVX2),
    IMPL("avx2x2", WriteBase10AVX2Batched),
    IMPL("sse_v2", WriteBase10SSEv2),
    IMPL("to_chars", WriteBase10ToChars),
    IMPL("hybrid", WriteBase10Hybrid),
    IMPL("picallo", WriteBase10Picallo),
};
#undef IMPL

static Implementation const *find_impl(std::string_view name)
{
    for (auto const &impl : implementations) {
        if (impl.name == name) {
            return &impl;
        }
    }
    return nullptr;
}

static void list_implementations()
{
    std::cerr << "Available implementations:" << std::endl;
    for (auto const &impl : implementations) {
        std::cerr << "  " << impl.name << std::endl;
    }
}

// ============================================================================
// Test (fuzz against slow reference)
// ============================================================================

static std::optional<std::tuple<uint256_t, std::string, std::string>> fuzz(
    to_string_fn_t reference, to_string_fn_t tested, size_t iterations,
    size_t max_bits = 256)
{
    auto gen = random_generator::from_fixed();
    std::string ref_out, test_out;
    ref_out.reserve(100);
    test_out.reserve(100);

    for (size_t i = 0; i < iterations; i++) {
        uint64_t const bw = gen.next_i() % (max_bits + 1);
        uint256_t const x = gen.next_b<uint256_t>(bw);

        reference(x, ref_out);
        tested(x, test_out);
        if (ref_out != test_out) {
            return {{x, ref_out, test_out}};
        }
        ref_out.clear();
        test_out.clear();
    }
    return std::nullopt;
}

// ============================================================================
// Benchmark (use `time` to measure)
// ============================================================================

[[gnu::noinline]]
static size_t bench(to_string_fn_t fn, size_t iterations, size_t max_bits = 256)
{
    auto gen = random_generator::from_fixed();
    std::string buffer;
    buffer.reserve(100);
    size_t total_len = 0;
    for (size_t i = 0; i < iterations; i++) {
        uint64_t const bw = gen.next_i() % (max_bits + 1);
        uint256_t const x = gen.next_b<uint256_t>(bw);
        fn(x, buffer);
        total_len += buffer.size();
        buffer.clear();
    }
    return total_len;
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(char const *prog)
{
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << prog << " test  <impl> <iterations> [--bits N]"
              << std::endl;
    std::cerr << "  " << prog << " bench <impl> <iterations> [--bits N]"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --bits N  Cap random inputs to N bits (default: 256)"
              << std::endl;
    std::cerr << std::endl;
    list_implementations();
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string_view mode = argv[1];
    std::string_view impl_name = argv[2];

    size_t n_iterations = 0;
    std::sscanf(argv[3], "%lu", &n_iterations);

    size_t max_bits = 256;
    for (int i = 4; i < argc; i++) {
        if (std::string_view(argv[i]) == "--bits" && i + 1 < argc) {
            std::sscanf(argv[++i], "%lu", &max_bits);
        }
    }

    auto const *impl = find_impl(impl_name);
    if (!impl) {
        std::cerr << "Unknown implementation: " << impl_name << std::endl;
        list_implementations();
        return 1;
    }

    if (mode == "test") {
        auto result =
            fuzz(to_string_slow<10>, impl->fn, n_iterations, max_bits);
        if (result) {
            auto const &[x, ref_out, test_out] = *result;
            std::cerr << "Discrepancy found!" << std::endl;
            std::cout << "\tx =               " << x.to_string(16) << std::endl;
            std::cout << "\tReference output: " << ref_out << std::endl;
            std::cout << "\tTested output:    " << test_out << std::endl;
            return 1;
        }
        std::cout << "All tests passed (" << n_iterations
                  << " random iterations)" << std::endl;
        return 0;
    }

    if (mode == "bench") {
        auto total = bench(impl->fn, n_iterations, max_bits);
        std::cout << "Total output length: " << total << std::endl;
        return 0;
    }

    std::cerr << "Unknown mode: " << mode << std::endl;
    print_usage(argv[0]);
    return 1;
}
