
def row_name(i):
    return str(i + 1)


def column_name(i):
    m_str = chr(65 + i % 26)
    n = i // 26
    n_str = '' if n == 0 else str(n)
    return m_str + n_str


class Visualization(object):
    pass
