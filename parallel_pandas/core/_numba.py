import numpy as np

try:
    from numba import njit, prange


    @njit
    def _pearson_corr(x, y):
        return np.corrcoef(x, y)[0, 1]


    @njit
    def calculate_rank(arr):
        n = len(arr)
        result = np.zeros(n, dtype=np.int64)
        indexes = np.argsort(arr)
        for i in range(n):
            result[indexes[i]] = i + 1

        return result


    @njit(fastmath=True)
    def _calculate_rank_dot_product(a, b):
        n = len(a)
        b_rank = calculate_rank(b)
        result = 0.0
        indexes = np.argsort(a)
        for i in range(n):
            result += (i + 1 - b_rank[indexes[i]]) * (i + 1 - b_rank[indexes[i]])

        return result


    @njit
    def _spearman_corr_with_rank(x, y):
        n = len(x)
        sum_d_squared = _calculate_rank_dot_product(x, y)
        correlation = 1 - (6 * sum_d_squared) / (n * (n ** 2 - 1))

        return correlation


    @njit(parallel=True)
    def _parallel_spearman_corr(mat, min_periods):
        n = mat.shape[0]
        c = mat.shape[1]
        rows, cols = np.triu_indices(c, k=1)
        corr_mat = np.zeros((c, c), dtype=np.float64)
        ranks = np.zeros((n, c), dtype=np.int64)
        isnan_flags = np.zeros(c, dtype=np.bool_)
        isnan = np.zeros((n, c), dtype=np.bool_)
        for i in prange(c):
            isnan[:, i] = np.isnan(mat[:, i])
            if isnan[:, i].any():
                isnan_flags[i] = True
            #
            else:
                ranks[:, i] = calculate_rank(mat[:, i])
        for i in prange(len(rows)):
            row, col = rows[i], cols[i]
            if isnan_flags[row] or isnan_flags[col]:
                valid = ~isnan[:, row] & ~isnan[:, col]
                if valid.sum() < min_periods:
                    corr_mat[row, col] = np.nan
                else:
                    corr_mat[row, col] = _spearman_corr_with_rank(mat[valid, row], mat[valid, col])
            else:
                d = ranks[:, row] - ranks[:, col]
                sum_d_squared = np.sum(d * d)
                corr_mat[row, col] = 1 - (6 * sum_d_squared) / (n * (n ** 2 - 1))
        corr_mat = corr_mat + corr_mat.transpose()
        np.fill_diagonal(corr_mat, 1)
        return corr_mat


    @njit(parallel=True)
    def _parallel_pearson_corr(mat, min_periods):
        n = mat.shape[0]
        c = mat.shape[1]
        rows, cols = np.triu_indices(c, k=1)
        corr_mat = np.zeros((mat.shape[1], mat.shape[1]), dtype=np.float64)
        isnan = np.isnan(mat)
        isnan_flags = isnan.sum(axis=0)
        sums = np.sum(mat, axis=0)
        sum_squareds = np.sum((mat * mat), axis=0)
        for i in prange(len(rows)):
            row, col = rows[i], cols[i]
            if isnan_flags[row] or isnan_flags[col]:
                valid = ~isnan[:, row] & ~isnan[:, col]
                if valid.sum() < min_periods:
                    corr_mat[row, col] = np.nan
                else:
                    corr_mat[row, col] = _pearson_corr(mat[valid, row], mat[valid, col])
            else:
                sum_xy = np.sum(mat[:, row] * mat[:, col])
                numerator = n * sum_xy - sums[row] * sums[col]
                denominator = np.sqrt(
                    (n * sum_squareds[row] - sums[row] * sums[row]) * (n * sum_squareds[col] - sums[col] * sums[col]))
                corr_mat[row, col] = numerator / denominator if denominator != 0 else 0
        corr_mat = corr_mat + corr_mat.transpose()
        np.fill_diagonal(corr_mat, 1)
        return corr_mat


    @njit(fastmath=True)
    def _kendall_tau(x, y):
        concordant = 0
        discordant = 0
        n = len(x)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Проверка согласованности или расхождения порядка
                if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    concordant += 1
                elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                    discordant += 1
        # Вычисление корреляции Кендалла
        tau = (concordant - discordant) / np.sqrt((concordant + discordant) * (n * (n - 1) / 2))

        return tau


    @njit(parallel=True)
    def _parallel_kendall(mat, min_periods):
        n = mat.shape[0]
        c = mat.shape[1]
        rows, cols = np.triu_indices(c, k=1)
        corr_mat = np.zeros((c, c), dtype=np.float64)
        isnan = np.isnan(mat)
        isnan_flags = isnan.sum(axis=0)
        for i in prange(len(rows)):
            row, col = rows[i], cols[i]
            a = mat[:, row]
            b = mat[:, col]
            if isnan_flags[row] or isnan_flags[col]:
                valid = ~isnan[:, row] & ~isnan[:, col]
                if valid.sum() < min_periods:
                    corr_mat[row, col] = np.nan
                else:
                    corr_mat[row, col] = _kendall_tau(a[valid], b[valid])
            else:
                corr_mat[row, col] = _kendall_tau(a, b)
        corr_mat = corr_mat + corr_mat.transpose()
        np.fill_diagonal(corr_mat, 1)
        return corr_mat


    @njit
    def _do_parallel_corr(mat, method, min_periods):
        if method == 'pearson':
            return _parallel_pearson_corr(mat, min_periods)
        elif method == 'spearman':
            return _parallel_spearman_corr(mat, min_periods)
        elif method == 'kendall':
            return _parallel_kendall(mat, min_periods)
        else:
            raise ValueError(f'Unknown method {method}')

except ImportError as e:
    def _do_parallel_corr(*args, **kwargs):
        raise ImportError('Numba not installed. Please install numba to use parallel pandas with numba engine.')
