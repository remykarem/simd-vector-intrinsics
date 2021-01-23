#include <stdio.h>
#include <immintrin.h>

void add_simple();
void sub_simple();
void add_simple_taketurns();
void add_simple_aligned();
void fma_simple();

void add_simple()
{
    float a[4] = {0.6, 0.7, 0.3, 0.1};
    float b[4] = {0.8, 0.2, 0.1, 0.3};
    float c[4];

    __m128 vec_a, vec_b, vec_c;

    vec_a = _mm_loadu_ps(a);
    vec_b = _mm_loadu_ps(b);
    vec_c = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(c, vec_c);

    for (int i = 0; i < 4; i++)
        printf("%f\n", c[i]);
}
void sub_simple()
{
    float a[4] = {0.6, 0.7, 0.3, 0.1};
    float b[4] = {0.8, 0.2, 0.1, 0.3};
    float c[4];

    __m128 vec_a, vec_b, vec_c;

    vec_a = _mm_loadu_ps(a);
    vec_b = _mm_loadu_ps(b);
    vec_c = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(c, vec_c);

    for (int i = 0; i < 4; i++)
        printf("%f\n", c[i]);
}

void add_simple_aligned()
{
    float a[4] __attribute__((aligned(32))) = {0.6, 0.7, 0.3, 0.1};
    float b[4] __attribute__((aligned(32))) = {0.8, 0.2, 0.1, 0.3};
    float c[4] __attribute__((aligned(32)));

    __m128 vec_a, vec_b, vec_c;

    vec_a = _mm_load_ps(a);
    vec_b = _mm_load_ps(b);
    vec_c = _mm_add_ps(vec_a, vec_b);
    _mm_store_ps(c, vec_c);

    for (int i = 0; i < 4; i++)
        printf("%f\n", c[i]);
}

void add_simple_taketurns()
{
    float a[8] = {0.6, 0.7, 0.3, 0.1, 0.0, 0.7, 0.3, 8.1};
    float b[8] = {0.8, 0.2, 0.1, 0.3, 0.6, 5.0, 9.3, 8.1};
    float c[8];

    __m128 vec_a, vec_b, vec_c;

    for (int i = 0; i < 8 / 4; i++)
    {
        vec_a = _mm_loadu_ps(a + i * 4);
        vec_b = _mm_loadu_ps(b + i * 4);
        vec_c = _mm_add_ps(vec_a, vec_b);
        _mm_storeu_ps(c + i * 4, vec_c);
    }

    for (int i = 0; i < 8; i++)
        printf("%f\n", c[i]);
}

void fma_simple()
{
    float a[8] = {0.0, 2.0, 3.0, 100.0};
    float b[8] = {0.6, 0.7, 0.3, 0.1};
    float c[8] = {0.8, 0.2, 0.1, 0.3};
    float d[8];

    __m128 vec_a, vec_b, vec_c, vec_d;

    vec_a = _mm_loadu_ps(a);
    vec_b = _mm_loadu_ps(b);
    vec_c = _mm_loadu_ps(c);
    vec_d = _mm_fmadd_ps(vec_a, vec_b, vec_c);
    _mm_storeu_ps(d, vec_d);

    for (int i = 0; i < 4; i++)
        printf("%f\n", d[i]);
}

int main()
{
    printf("--1--\n");
    add_simple();
    printf("--2--\n");
    sub_simple();
    printf("--3--\n");
    add_simple_taketurns();
    printf("--4--\n");
    add_simple_aligned();
    printf("--5--\n");
    fma_simple();
    
    return 0;
}
