from math import floor
from scipy.special import erfc, gammaincc, hyp1f1
from scipy.stats import norm
import numpy as np

class Tests:
    @staticmethod
    def monobit_test(binary_data):
        """
        Выполняет побитный частотный тест на двоичных данных.
        :param bin_data: Двоичная строка для тестирования
        :return: p-значение из теста
        """
        count = 0
        # Если символ равен 0, вычитаем 1, иначе прибавляем 1
        for char in binary_data:
            if char == '0':
                count -= 1
            else:
                count += 1
        # Вычисляем p-значение
        sobs = count / np.sqrt(len(binary_data))
        p_val = erfc(abs(sobs) / np.sqrt(2))
        return p_val

    @staticmethod
    def block_frequency_test(binary_data, block_size = 128):
        """
        Выполняет блочный частотный тест на двоичных данных.
        :param binary_data: Двоичная строка для тестирования
        :param block_size: Размер блока
        :return: p-значение из теста
        """
        length_of_bit_string = len(binary_data)


        if length_of_bit_string < block_size:
            block_size = length_of_bit_string

        number_of_blocks = floor(length_of_bit_string / block_size)

        if number_of_blocks == 1:
            return Tests.monobit_test(binary_data[0:block_size])

        block_start = 0
        block_end = block_size
        proportion_sum = 0.0

        for counter in range(number_of_blocks):
            block_data = binary_data[block_start:block_end]

            one_count = 0
            for bit in block_data:
                if bit == '1':
                    one_count += 1
            pi = one_count / block_size

            proportion_sum += pow(pi - 0.5, 2.0)

            block_start += block_size
            block_end += block_size

        result = 4.0 * block_size * proportion_sum

        p_value = gammaincc(number_of_blocks / 2, result / 2)

        return (p_value)

    @staticmethod
    def longest_run_ones_in_block_test(binary_data):
        """
        Выполняет тест на самую длинную последовательность единиц в блоке на двоичных данных.
        :param bin_data: Двоичная строка для тестирования
        :param block_size: Размер блока
        :return: p-значение из теста
        """
        length_of_binary_data = len(binary_data)

        k = 3
        m = 8
        v_values = [1, 2, 3, 4]
        pi_values = [0.21484375, 0.3671875, 0.23046875, 0.1875]

        number_of_blocks = floor(length_of_binary_data / m)
        block_start = 0
        block_end = m
        xObs = 0

        frequencies = np.zeros(k + 1)

        for count in range(number_of_blocks):
            block_data = binary_data[block_start:block_end]
            max_run_count = 0
            run_count = 0

            for bit in block_data:
                if bit == '1':
                    run_count += 1
                    max_run_count = max(max_run_count, run_count)
                else:
                    max_run_count = max(max_run_count, run_count)
                    run_count = 0

            max(max_run_count, run_count)

            if max_run_count < v_values[0]:
                frequencies[0] += 1
            for j in range(k):
                if max_run_count == v_values[j]:
                    frequencies[j] += 1
            if max_run_count > v_values[k - 1]:
                frequencies[k] += 1

            block_start += m
            block_end += m

        for count in range(len(frequencies)):
            xObs += pow((frequencies[count] - (number_of_blocks * pi_values[count])), 2.0) / (
                    number_of_blocks * pi_values[count])

        p_value = gammaincc(float(k / 2), float(xObs / 2))
        return p_value
    
    @staticmethod
    def non_overlapping_test(binary_data, template_pattern='000000001', block=8):
        """
        Note that this description is taken from the NIST documentation [1]
        [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
        The focus of this test is the number of occurrences of pre-specified target strings. The purpose of this
        test is to detect generators that produce too many occurrences of a given non-periodic (aperiodic) pattern.
        For this test and for the Overlapping Template Matching test of Section 2.8, an m-bit window is used to
        search for a specific m-bit pattern. If the pattern is not found, the window slides one bit position. If the
        pattern is found, the window is reset to the bit after the found pattern, and the search resumes.
        :param      binary_data:        The seuqnce of bit being tested
        :param      template_pattern:   The pattern to match to
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :param      block               The number of independent blocks. Has been fixed at 8 in the test code.
        :return:    (p_value, bool)     A tuple which contain the p_value and result of frequency_test(True or False)
        """

        length_of_binary = len(binary_data)
        pattern_size = len(template_pattern)
        block_size = floor(length_of_binary / block)
        pattern_counts = np.zeros(block)

        # For each block in the data
        for count in range(block):
            block_start = count * block_size
            block_end = block_start + block_size
            block_data = binary_data[block_start:block_end]
            # Count the number of pattern hits
            inner_count = 0
            while inner_count < block_size:
                sub_block = block_data[inner_count:inner_count+pattern_size]
                if sub_block == template_pattern:
                    pattern_counts[count] += 1
                    inner_count += pattern_size
                else:
                    inner_count += 1

            # Calculate the theoretical mean and variance
            # Mean - µ = (M-m+1)/2m
            mean = (block_size - pattern_size + 1) / pow(2, pattern_size)
            # Variance - σ2 = M((1/pow(2,m)) - ((2m -1)/pow(2, 2m)))
            variance = block_size * ((1 / pow(2, pattern_size)) - (((2 * pattern_size) - 1) / (pow(2, pattern_size * 2))))

        # Calculate the xObs Squared statistic for these pattern matches
        xObs = 0
        for count in range(block):
            xObs += pow((pattern_counts[count] - mean), 2.0) / variance

        # Calculate and return the p value statistic
        p_value = gammaincc((block / 2), (xObs / 2))
        return (p_value)

    @staticmethod
    def overlapping_patterns(binary_data, pattern_size=16, block_size=128):
        """
        Note that this description is taken from the NIST documentation [1]
        [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
        The focus of the Overlapping Template Matching test is the number of occurrences of pre-specified target
        strings. Both this test and the Non-overlapping Template Matching test of Section 2.7 use an m-bit
        window to search for a specific m-bit pattern. As with the test in Section 2.7, if the pattern is not found,
        the window slides one bit position. The difference between this test and the test in Section 2.7 is that
        when the pattern is found, the window slides only one bit before resuming the search.

        :param      binary_data:    a binary string
        :param      verbose         True to display the debug messgae, False to turn off debug message
        :param      pattern_size:   the length of the pattern
        :param      block_size:     the length of the block
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)
        """
        length_of_binary_data = len(binary_data)
        pattern = ''
        for count in range(pattern_size):
            pattern += '1'

        number_of_block = floor(length_of_binary_data / block_size)

        # λ = (M-m+1)/pow(2, m)
        lambda_val = float(block_size - pattern_size + 1) / pow(2, pattern_size)
        # η = λ/2
        eta = lambda_val / 2.0

        pi = [Tests.get_prob(i, eta) for i in range(5)]
        diff = float(np.array(pi).sum())
        pi.append(1.0 - diff)

        pattern_counts = np.zeros(6)
        for i in range(number_of_block):
            block_start = i * block_size
            block_end = block_start + block_size
            block_data = binary_data[block_start:block_end]
            # Count the number of pattern hits
            pattern_count = 0
            j = 0
            while j < block_size:
                sub_block = block_data[j:j + pattern_size]
                if sub_block == pattern:
                    pattern_count += 1
                j += 1
            if pattern_count <= 4:
                pattern_counts[pattern_count] += 1
            else:
                pattern_counts[5] += 1

        xObs = 0.0
        for i in range(len(pattern_counts)):
            xObs += pow(pattern_counts[i] - number_of_block * pi[i], 2.0) / (number_of_block * pi[i])

        p_value = gammaincc(5.0 / 2.0, xObs / 2.0)
        return (p_value)

    @staticmethod
    def get_prob(u, x):
        out = 1.0 * np.exp(-x)
        if u != 0:
            out = 1.0 * x * np.exp(2 * -x) * (2 ** -u) * hyp1f1(u + 1, 2, x)
        return out
    
    @staticmethod
    def statistical_test(binary_data):
        """
        Note that this description is taken from the NIST documentation [1]
        [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
        The focus of this test is the number of bits between matching patterns (a measure that is related to the
        length of a compressed sequence). The purpose of the test is to detect whether or not the sequence can be
        significantly compressed without loss of information. A significantly compressible sequence is considered
        to be non-random. **This test is always skipped because the requirements on the lengths of the binary
        strings are too high i.e. there have not been enough trading days to meet the requirements.

        :param      binary_data:    a binary string
        :param      verbose             True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)
        """
        length_of_binary_data = len(binary_data)
        pattern_size = 5
        if length_of_binary_data >= 387840:
            pattern_size = 6
        if length_of_binary_data >= 904960:
            pattern_size = 7
        if length_of_binary_data >= 2068480:
            pattern_size = 8
        if length_of_binary_data >= 4654080:
            pattern_size = 9
        if length_of_binary_data >= 10342400:
            pattern_size = 10
        if length_of_binary_data >= 22753280:
            pattern_size = 11
        if length_of_binary_data >= 49643520:
            pattern_size = 12
        if length_of_binary_data >= 107560960:
            pattern_size = 13
        if length_of_binary_data >= 231669760:
            pattern_size = 14
        if length_of_binary_data >= 496435200:
            pattern_size = 15
        if length_of_binary_data >= 1059061760:
            pattern_size = 16

        if 5 < pattern_size < 16:
            # Create the biggest binary string of length pattern_size
            ones = ""
            for i in range(pattern_size):
                ones += "1"

            # How long the state list should be
            num_ints = int(ones, 2)
            vobs = np.zeros(num_ints + 1)

            # Keeps track of the blocks, and whether were are initializing or summing
            num_blocks = floor(length_of_binary_data / pattern_size)
            # Q = 10 * pow(2, pattern_size)
            init_bits = 10 * pow(2, pattern_size)

            test_bits = num_blocks - init_bits

            c = 0.7 - 0.8 / pattern_size + (4 + 32 / pattern_size) * pow(test_bits, -3 / pattern_size) / 15
            variance = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
            expected = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656, 8.1764248, 9.1723243,
                        10.170032, 11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
            sigma = c * np.sqrt(variance[pattern_size] / test_bits)

            cumsum = 0.0

            for i in range(num_blocks):
                block_start = i * pattern_size
                block_end = block_start + pattern_size
                block_data = binary_data[block_start: block_end]

                int_rep = int(block_data, 2)

                if i < init_bits:
                    vobs[int_rep] = i + 1
                else:
                    initial = vobs[int_rep]
                    vobs[int_rep] = i + 1
                    cumsum += np.log(i - initial + 1, 2)

            phi = float(cumsum / test_bits)
            stat = abs(phi - expected[pattern_size]) / (float(np.sqrt(2)) * sigma)

            p_value = erfc(stat)

            return (p_value, (p_value>=0.01))
        else:
            return (-1.0, False)
        
    @staticmethod
    def linear_complexity_test(binary_data, block_size=500):
        """
        Note that this description is taken from the NIST documentation [1]
        [1] http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf
        The focus of this test is the length of a linear feedback shift register (LFSR). The purpose of this test is to
        determine whether or not the sequence is complex enough to be considered random. Random sequences are
        characterized by longer LFSRs. An LFSR that is too short implies non-randomness.

        :param      binary_data:    a binary string
        :param      verbose         True to display the debug messgae, False to turn off debug message
        :param      block_size:     Size of the block
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)

        """

        length_of_binary_data = len(binary_data)

        # The number of degrees of freedom;
        # K = 6 has been hard coded into the test.
        degree_of_freedom = 6

        #  π0 = 0.010417, π1 = 0.03125, π2 = 0.125, π3 = 0.5, π4 = 0.25, π5 = 0.0625, π6 = 0.020833
        #  are the probabilities computed by the equations in Section 3.10
        pi = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

        t2 = (block_size / 3.0 + 2.0 / 9) / 2 ** block_size
        mean = 0.5 * block_size + (1.0 / 36) * (9 + (-1) ** (block_size + 1)) - t2

        number_of_block = int(length_of_binary_data / block_size)

        if number_of_block > 1:
            block_end = block_size
            block_start = 0
            blocks = []
            for i in range(number_of_block):
                blocks.append(binary_data[block_start:block_end])
                block_start += block_size
                block_end += block_size

            complexities = []
            for block in blocks:
                complexities.append(Tests.berlekamp_massey_algorithm(block))

            t = ([-1.0 * (((-1) ** block_size) * (chunk - mean) + 2.0 / 9) for chunk in complexities])
            vg = np.histogram(t, bins=[-9999999999, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9999999999])[0][::-1]
            im = ([((vg[ii] - number_of_block * pi[ii]) ** 2) / (number_of_block * pi[ii]) for ii in range(7)])

            xObs = 0.0
            for i in range(len(pi)):
                xObs += im[i]

            # P-Value = igamc(K/2, xObs/2)
            p_value = gammaincc(degree_of_freedom / 2.0, xObs / 2.0)

            return (p_value, (p_value >= 0.01))
        else:
            return (-1.0, False)
        
    @staticmethod
    def berlekamp_massey_algorithm(block_data):
        """
        An implementation of the Berlekamp Massey Algorithm. Taken from Wikipedia [1]
        [1] - https://en.wikipedia.org/wiki/Berlekamp-Massey_algorithm
        The Berlekamp–Massey algorithm is an algorithm that will find the shortest linear feedback shift register (LFSR)
        for a given binary output sequence. The algorithm will also find the minimal polynomial of a linearly recurrent
        sequence in an arbitrary field. The field requirement means that the Berlekamp–Massey algorithm requires all
        non-zero elements to have a multiplicative inverse.
        :param block_data:
        :return:
        """
        n = len(block_data)
        c = np.zeros(n)
        b = np.zeros(n)
        c[0], b[0] = 1, 1
        l, m, i = 0, -1, 0
        int_data = [int(el) for el in block_data]
        while i < n:
            v = int_data[(i - l):i]
            v = v[::-1]
            cc = c[1:l + 1]
            d = (int_data[i] + np.dot(v, cc)) % 2
            if d == 1:
                temp = np.copy(c)
                p = np.zeros(n)
                for j in range(0, l):
                    if b[j] == 1:
                        p[j + i - m] = 1
                c = (c + p) % 2
                if l <= 0.5 * i:
                    l = i + 1 - l
                    m = i
                    b = temp
            i += 1
        return l
    
    @staticmethod
    def approximate_entropy_test(binary_data, pattern_length=10):
        """
        from the NIST documentation http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf

        As with the Serial test of Section 2.11, the focus of this test is the frequency of all possible
        overlapping m-bit patterns across the entire sequence. The purpose of the test is to compare
        the frequency of overlapping blocks of two consecutive/adjacent lengths (m and m+1) against the
        expected result for a random sequence.

        :param      binary_data:        a binary string
        :param      verbose             True to display the debug message, False to turn off debug message
        :param      pattern_length:     the length of the pattern (m)
        :return:    ((p_value1, bool), (p_value2, bool)) A tuple which contain the p_value and result of serial_test(True or False)
        """
        length_of_binary_data = len(binary_data)

        # Augment the n-bit sequence to create n overlapping m-bit sequences by appending m-1 bits
        # from the beginning of the sequence to the end of the sequence.
        # NOTE: documentation says m-1 bits but that doesnt make sense, or work.
        binary_data += binary_data[:pattern_length + 1:]

        # Get max length one patterns for m, m-1, m-2
        max_pattern = ''
        for i in range(pattern_length + 2):
            max_pattern += '1'

        # Keep track of each pattern's frequency (how often it appears)
        vobs_01 = np.zeros(int(max_pattern[0:pattern_length:], 2) + 1)
        vobs_02 = np.zeros(int(max_pattern[0:pattern_length + 1:], 2) + 1)

        for i in range(length_of_binary_data):
            # Work out what pattern is observed
            vobs_01[int(binary_data[i:i + pattern_length:], 2)] += 1
            vobs_02[int(binary_data[i:i + pattern_length + 1:], 2)] += 1

        # Calculate the test statistics and p values
        vobs = [vobs_01, vobs_02]

        sums = np.zeros(2)
        for i in range(2):
            for j in range(len(vobs[i])):
                if vobs[i][j] > 0:
                    sums[i] += vobs[i][j] * np.log(vobs[i][j] / length_of_binary_data)
        sums /= length_of_binary_data
        ape = sums[0] - sums[1]

        xObs = 2.0 * length_of_binary_data * (np.log(2) - ape)

        p_value = gammaincc(pow(2, pattern_length - 1), xObs / 2.0)

        return (p_value, (p_value >= 0.01))
    
    @staticmethod
    def cumulative_sums_test(binary_data:str, mode=0, verbose=False):
        """
        from the NIST documentation http://csrc.nist.gov/publications/nistpubs/800-22-rev1a/SP800-22rev1a.pdf

        The focus of this test is the maximal excursion (from zero) of the random walk defined by the cumulative sum of
        adjusted (-1, +1) digits in the sequence. The purpose of the test is to determine whether the cumulative sum of
        the partial sequences occurring in the tested sequence is too large or too small relative to the expected
        behavior of that cumulative sum for random sequences. This cumulative sum may be considered as a random walk.
        For a random sequence, the excursions of the random walk should be near zero. For certain types of non-random
        sequences, the excursions of this random walk from zero will be large.

        :param      binary_data:    a binary string
        :param      mode            A switch for applying the test either forward through the input sequence (mode = 0)
                                    or backward through the sequence (mode = 1).
        :param      verbose         True to display the debug messgae, False to turn off debug message
        :return:    (p_value, bool) A tuple which contain the p_value and result of frequency_test(True or False)

        """

        length_of_binary_data = len(binary_data)
        counts = np.zeros(length_of_binary_data)

        # Determine whether forward or backward data
        if not mode == 0:
            binary_data = binary_data[::-1]

        counter = 0
        for char in binary_data:
            sub = 1
            if char == '0':
                sub = -1
            if counter > 0:
                counts[counter] = counts[counter -1] + sub
            else:
                counts[counter] = sub

            counter += 1
        # Compute the test statistic z =max1≤k≤n|Sk|, where max1≤k≤n|Sk| is the largest of the
        # absolute values of the partial sums Sk.
        abs_max = max(abs(counts))

        start = int(floor(0.25 * floor(-length_of_binary_data / abs_max + 1)))
        end = int(floor(0.25 * floor(length_of_binary_data / abs_max - 1)))

        terms_one = []
        for k in range(start, end + 1):
            sub = norm.cdf((4 * k - 1) * abs_max / np.sqrt(length_of_binary_data))
            terms_one.append(norm.cdf((4 * k + 1) * abs_max / np.sqrt(length_of_binary_data)) - sub)

        start = int(floor(0.25 * floor(-length_of_binary_data / abs_max - 3)))
        end = int(floor(0.25 * floor(length_of_binary_data / abs_max) - 1))

        terms_two = []
        for k in range(start, end + 1):
            sub = norm.cdf((4 * k + 1) * abs_max / np.sqrt(length_of_binary_data))
            terms_two.append(norm.cdf((4 * k + 3) * abs_max / np.sqrt(length_of_binary_data)) - sub)

        p_value = 1.0 - sum(np.array(terms_one))
        p_value += sum(np.array(terms_two))

        if verbose:
            print('Cumulative Sums Test DEBUG BEGIN:')
            print("\tLength of input:\t", length_of_binary_data)
            print('\tMode:\t\t\t\t', mode)
            print('\tValue of z:\t\t\t', abs_max)
            print('\tP-Value:\t\t\t', p_value)
            print('DEBUG END.')

        return (p_value, (p_value >= 0.01))