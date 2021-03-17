// Hungarian (Kuhn-Munkres) algorithm's C++17 header-only implementation.
// Based on this implementation:
// https://github.com/mcximing/hungarian-algorithm-cpp

// By Sergey Evsegneev, 2021
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#ifdef ENABLE_PMR
#include <memory_resource>
#endif
#include <numeric>
#include <vector>

namespace hungarian_alg
{
template<class T, bool row_major>
class solver final
{
    using bool_t = int_fast8_t;
#ifdef ENABLE_PMR
    using bool_vector = std::pmr::vector<bool_t>;
#else
    using bool_vector = std::vector<bool_t>;
#endif

    T* const data;
    const T epsilon;
    const size_t rows;
    const size_t cols;
    const size_t stride;
    const size_t ndims;
    bool_vector covered_cols;
    bool_vector covered_rows;
    bool_vector star_matrix;
    bool_vector prime_matrix;
    bool_vector new_star_matrix;

public:
    using allocator_type = typename bool_vector::allocator_type;

    solver(T* _data, size_t _rows, size_t _cols, size_t _stride, const allocator_type& _allocator)
        : data(_data)
        , epsilon([=]
            {
                T result = {};
                if constexpr (!std::is_integral_v<T>)
                {
                    if (_stride == 0)
                        result = *std::max_element(_data, _data + _cols * _rows);
                    else
                        for (size_t i = 0; i < (row_major ? _rows : _cols); ++i)
                            result = std::max(result, *std::max_element(_data + i * _stride, _data + (i + 1) * _stride));
                    result *= std::numeric_limits<T>::epsilon();
                }
                return result;
            }())
        , rows(_rows)
        , cols(_cols)
        , stride(_stride != 0 ? _stride : row_major ? _cols : _rows)
        , ndims(std::min(_rows, _cols))
        , covered_cols(_cols, 0, _allocator)
        , covered_rows(_rows, 0, _allocator)
        , star_matrix(_cols * _rows, 0, _allocator)
        , prime_matrix(star_matrix.size(), 0, _allocator)
        , new_star_matrix(star_matrix.size(), 0, _allocator)
    {}

    solver(const solver&) = delete;

    [[nodiscard]] static size_t required_memory(size_t cols, size_t rows) noexcept
    {
        return sizeof(rows + cols + 3 * rows * cols) + (size_t)512;
    }

    void solve(std::vector<size_t>& assignment)
    {
        for (step1(); std::accumulate(covered_cols.begin(), covered_cols.end(), size_t{}) != ndims; step2())
            while (step3())
                step5();
        get_assignment(assignment);
    }

private:
    [[nodiscard]] bool is_zero(const T& val) const noexcept
    {
        if constexpr (std::is_integral_v<T>)
            return val == T{};
        else
            return val <= epsilon;
    }

    [[nodiscard]] T& at(size_t col, size_t row) noexcept
    {
        return data[row_major ? (row * stride + col) : (col * stride + row)];
    }

    void get_assignment(std::vector<size_t>& assignment) const
    {
        if (assignment.size() != rows)
        {
            assignment.clear();
            assignment.resize(rows);
        }
        auto it_end = star_matrix.begin();
        for (size_t& x: assignment)
        {
            const auto it_begin = it_end;
            std::advance(it_end, static_cast<ptrdiff_t>(cols));
            const auto it = std::find(it_begin, it_end, true);
            x = static_cast<size_t>(std::distance(it_begin, it));
        }
    }

    void step1() noexcept
    {
        if (rows <= cols)
        {
            for (size_t row = 0; row < rows; ++row)
            {
                T h = at(0, row);
                size_t col;
                for (col = 1; col < cols; ++col)
                    h = std::min(h, at(col, row));
                for (col = 0; col < cols; ++col)
                    if (is_zero(at(col, row) -= h) && !covered_cols[col])
                    {
                        star_matrix[row * cols + col] = true;
                        covered_cols[col++] = true;
                        break;
                    }
                for (; col < cols; ++col)
                    at(col, row) -= h;
            }
        }
        else
        {
            for (size_t col = 0; col < cols; ++col)
            {
                T h = at(col, 0);
                size_t row;
                for (row = 1; row < rows; ++row)
                    h = std::min(h, at(col, row));
                for (row = 0; row < rows; ++row)
                    if (is_zero(at(col, row) -= h) && !covered_rows[row])
                    {
                        star_matrix[row * cols + col] = true;
                        covered_cols[col] = true;
                        covered_rows[row++] = true;
                        break;
                    }
                for (; row < rows; ++row)
                    at(col, row) -= h;
            }
            std::fill(covered_rows.begin(), covered_rows.end(), false);
        }
    }

    void step2() noexcept
    {
        for (size_t col = 0; col < cols; ++col)
            for (size_t row = 0; row < rows; ++row)
                if (star_matrix[row * cols + col])
                {
                    covered_cols[col] = true;
                    break;
                }
    }

    [[nodiscard]] size_t find_row(const bool_vector& v, size_t col) const noexcept
    {
        for (size_t row = 0; row < rows; ++row)
            if(v[row * cols + col])
                return row;
        return rows;
    }

    [[nodiscard]] size_t find_col(const bool_vector& v, size_t row) const noexcept
    {
        for (size_t col = 0; col < cols; ++col)
            if (v[row * cols + col])
                return col;
        return cols;
    }

    bool step3() noexcept
    {
        bool zeroes_found;
        do
        {
            zeroes_found = false;
            for (size_t col = 0; col < cols; ++col)
                if (!covered_cols[col])
                    for (size_t row = 0; row < rows; ++row)
                        if (!covered_rows[row] && is_zero(at(col, row)))
                        {
                            prime_matrix[row * cols + col] = true;
                            size_t star_col = find_col(star_matrix, row);
                            if (star_col == cols)
                            {
                                step4(row, col);
                                return false;
                            }
                            else
                            {
                                covered_rows[row] = true;
                                covered_cols[star_col] = false;
                                zeroes_found = true;
                                break;
                            }
                        }
        } while (zeroes_found);
        return true;
    }

    void step4(size_t row, size_t col) noexcept
    {
        std::copy(star_matrix.begin(), star_matrix.end(), new_star_matrix.begin());
        new_star_matrix[row * cols + col] = true;
        size_t star_col = col;
        size_t star_row = find_row(star_matrix, star_col);
        while (star_row < rows)
        {
            new_star_matrix[star_row * cols + star_col] = false;
            star_col = find_col(prime_matrix, star_row);
            new_star_matrix[star_row * cols + star_col] = true;
            star_row = find_row(star_matrix, star_col);
        }
        std::fill(prime_matrix.begin(), prime_matrix.end(), false);
        std::fill(covered_rows.begin(), covered_rows.end(), false);
        star_matrix.swap(new_star_matrix);
    }

    void step5() noexcept
    {
        double h = std::numeric_limits<double>::max();
        for (size_t row = 0; row < rows; ++row)
            if (!covered_rows[row])
                for (size_t col = 0; col < cols; ++col)
                    if (!covered_cols[col])
                        h = std::min(h, at(col, row));

        for (size_t row = 0; row < rows; ++row)
            if(covered_rows[row])
                for (size_t col = 0; col < cols; ++col)
                    at(col, row) += h;

        for (size_t col = 0; col < cols; ++col)
            if(!covered_cols[col])
                for (size_t row = 0; row < rows; ++row)
                    at(col, row) -= h;
    }
};

template<bool row_major=true, class T>
void solve(std::vector<size_t>& assignment, T* distance_matrix, size_t cols, size_t rows, size_t stride = 0)
{
    static_assert(std::is_arithmetic_v<T>);
    using TT = std::remove_cv_t<T>;
    using solver_t = solver<TT, row_major>;
    TT* data;
    if (stride == (row_major ? cols : rows))
        stride = 0;
#ifdef ENABLE_PMR
    const size_t required_memory = solver_t::required_memory(cols, rows) + (std::is_const_v<T> ? sizeof(T) * cols * rows : 0);
    std::pmr::monotonic_buffer_resource mem(required_memory);
    typename solver_t::allocator_type alloc{ &mem };
#else
    typename solver_t::allocator_type alloc{ };
    std::vector<TT> copied_data;
#endif
    if constexpr (std::is_const_v<T>)
    {
#ifdef ENABLE_PMR
        data = std::pmr::polymorphic_allocator<TT>(alloc).allocate(cols * rows);
#else
        copied_data.resize(cols * rows);
        data = std::data(copied_data);
#endif
        if (stride == 0)
            std::uninitialized_copy(distance_matrix, distance_matrix + cols * rows, data);
        else
            for (size_t i = 0; i < (row_major ? rows : cols); ++i)
                std::uninitialized_copy(distance_matrix + i * stride, distance_matrix + (i + 1) * stride, data);
    }
    else
        data = distance_matrix;
    solver_t s(data, rows, cols, stride, alloc);
    s.solve(assignment);
}

template<bool row_major = true, class T>
T solve_with_cost(std::vector<size_t>& assignment, const T* distance_matrix, size_t cols, size_t rows, size_t stride = 0)
{
    solve<row_major, const T>(assignment, distance_matrix, cols, rows, stride);
    T cost = {};
    stride = stride != 0 ? stride : row_major ? cols : rows;
    for (size_t row = 0; row < rows; ++row)
        if (assignment[row] != cols)
            cost += distance_matrix[row_major ? (row * stride + assignment[row]) : (assignment[row] * stride + row)];
    return cost;
}
}
