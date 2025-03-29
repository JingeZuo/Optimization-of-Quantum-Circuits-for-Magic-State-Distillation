from sympy import eye
import numpy as np
import math
from sympy import symbols, Poly, ZZ
#%%
def mod2(matrix):
    """Reduce a matrix to modulo 2"""
    return np.mod(matrix, 2)

def gf2_rref(matrix):
    """Compute the reduced row echelon form (RREF) of a matrix in GF(2)"""
    matrix = mod2(matrix)
    rows, cols = matrix.shape
    row, col = 0, 0

    while row < rows and col < cols:
        if matrix[row, col] == 0:
            for r in range(row + 1, rows):
                if matrix[r, col] == 1:
                    matrix[[row, r]] = matrix[[r, row]]
                    break
        if matrix[row, col] == 1:
            for r in range(rows):
                if r != row and matrix[r, col] == 1:
                    matrix[r] = mod2(matrix[r] + matrix[row])
            row += 1
        col += 1

    return matrix


def polynomial_to_circulant(poly, l):
    x = symbols('x')

    if not isinstance(poly, Poly):
        poly = Poly(poly, x, domain=ZZ)

    coeffs = poly.all_coeffs()

    # 系数需要从低次幂到高次幂排列
    coeffs.reverse()

    # Create a l \times l matrix
    circulant_matrix = np.zeros((l, l), dtype=int)

    # Fill the elements to get the circulant of  the "poly"
    for i in range(l):
        a1 = np.pad(coeffs, (0, l - len(coeffs)), 'constant')
        circulant_matrix[i] = row = np.roll(a1, i)

    return circulant_matrix
def kernel_gf2(matrix):
    """Compute the kernel of a matrix in GF(2)"""
    matrix_rref = gf2_rref(matrix)
    rows, cols = matrix_rref.shape
    pivot_cols = []

    for r in range(rows):
        for c in range(cols):
            if matrix_rref[r, c] == 1:
                pivot_cols.append(c)
                break

    free_vars = [c for c in range(cols) if c not in pivot_cols]
    kernel_basis = []
    #print(free_vars)
    for free_var in free_vars:
        basis_vector = np.zeros(cols, dtype=int)
        basis_vector[free_var] = 1
        for pivot_col in pivot_cols:
            row = pivot_cols.index(pivot_col)
            if matrix_rref[row, free_var] == 1:
                basis_vector[pivot_col] = 1
        kernel_basis.append(basis_vector)

    return np.array(kernel_basis).T

def swap_columns(matrix, col1, col2):
    matrix[:, [col1, col2]] = matrix[:, [col2, col1]]
    return matrix
def swap_rows(matrix, row1, row2):
    matrix[[row1,row2],:] = matrix[[row2,row1],:]
    return matrix

def create_identity_at_top(matrix):
    """Ensure the first k rows and k columns form an identity matrix using column swaps"""
    rows, cols = matrix.shape
    P = np.eye(cols)

    for i in range(rows):
        if matrix[i, i] != 1:
            # Find a column with a 1 in the i-th row
            for j in range(i + 1, cols):
                if matrix[i, j] == 1:
                    swap_columns(matrix, i, j)
                    swap_rows(P, i, j)
                    break
    return matrix, P
def hamming_weight(vector):
    """计算一个向量的汉明权重"""
    return np.sum(vector != 0)

def min_hamming_weight(matrix):
    """计算矩阵中所有行向量的最小汉明权重"""
    # 计算每一行的汉明权重
    weights = np.apply_along_axis(hamming_weight, 1, matrix)
    # 返回最小的汉明权重
    return np.min(weights)
def max_hamming_weight(matrix):
    """计算矩阵中所有行向量的最小汉明权重"""
    # 计算每一行的汉明权重
    weights = np.apply_along_axis(hamming_weight, 1, matrix)
    # 返回最小的汉明权重
    return np.max(weights)
#%%
k1=4
H1=np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]])
d1=min_hamming_weight(kernel_gf2(H1).T)
H2=H1#经典码校验矩阵
r1=H1.shape[0]
r2=H2.shape[0]
n1=H1.shape[1]
n2=H2.shape[1]
E1=np.eye(r1)
E2=np.eye(r2)
En1=np.eye(n1)
En2=np.eye(n2)
G1=np.array([
    [1, 0, 0, 0, 0,1,1],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 1,1,0],
    [0, 0, 0, 1, 1,1,1]
], dtype=int)#经典码生成矩阵
G2=G1
HX=np.kron(H1,G2)
HZ=np.kron(G1,H2)#校验矩阵
LX=np.kron(En1,G2)
LZ=np.kron(G1,En2)#逻辑算符生成矩阵
XL=gf2_rref(np.vstack([LX,HX]))[:16]
ZL=gf2_rref(np.vstack([LZ,HZ]))[:16]#去除与稳定子线性相关的部分


JZA=np.array([[0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0],
              [1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
              [1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
              [1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0],
              [1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0],
              [1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
              [1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0],
              [1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0],
              [0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0],
              [0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0],
              [0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1]])#并行测量的逻辑算符

q=JZA.shape[0]

JZC=np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])#补空间
JZa=[]
JZc=[]
for i in range(JZA.shape[0]):
    a=np.zeros(49)
    for j in range(JZA.shape[1]):
        a=a+JZA[i,j]*XL[j]
    JZa.append(a)
JZa=np.array(JZa)
JZa=mod2(JZa)
for i in range(JZC.shape[0]):
    a=np.zeros(49)
    for j in range(JZC.shape[1]):
        a=a+JZC[i,j]*XL[j]
    JZc.append(a)
JZc=mod2(JZc)
JZa=np.array(JZa)
JZc=np.array(JZc)#逻辑量子比特形式转换为物理量子比特形式
zero_columns = []
for col in range(JZa.shape[1]):
    if np.all(ZL[:, col] == 0):  # 检查该列是否全为 0
        zero_columns.append(col)
if zero_columns:
    print(f"全 0 列的索引: {zero_columns}")
else:
    print("矩阵中不存在全 0 列")
nN=XL.shape[1]-len(zero_columns)#被测量算符的支撑区域

LogicalX_qs0 = JZa#J_{Z,A }
LogicalZ_qs0 = ZL
LogicalX_qs1 = JZc  # J_{Z,C }
LogicalX_qs = np.array(LogicalX_qs0.tolist())
LogicalZ_qs = np.array(LogicalZ_qs0.tolist())
LogicalX_qs1 = np.array(LogicalX_qs1.tolist())
non_zero_cols = [col_idx for col_idx in range(LogicalX_qs0.shape[1]) if
                 not all(row[col_idx] == 0 for row in LogicalX_qs)]
gg = len(non_zero_cols)

delelogicalX = LogicalX_qs[:, non_zero_cols]
delelogicalZ = LogicalZ_qs[:, non_zero_cols]
delelogicalX1 = np.array(delelogicalX.tolist())
delelogicalX2 = delelogicalX1[~np.all(delelogicalX1 == 0, axis=1)]  # 消除全为零元的行，这步操作可以没有，因为无全零行
f = delelogicalZ.shape[1]
H_Z = HZ # get the parity check matrix H_Z
filtered_matrix = H_Z[:, non_zero_cols]
filtered_matrix1 = np.array(filtered_matrix.tolist())
filtered_matrix2 = filtered_matrix1[~np.all(filtered_matrix1 == 0, axis=1)]  # get H_N
saved_rows = []
saved_rows1 = []
K = kernel_gf2(filtered_matrix2).T  # the kernel of H_N

for row in K:

    row_matrix = row.reshape(1, -1)

    result_vector = np.dot(row_matrix, delelogicalZ.T) % 2

    if np.any(result_vector != 0):
        saved_rows2 = [r.flatten() for r in saved_rows1]  # 转换为列表
        F = np.array(saved_rows)
        m4 = F.shape[0]
        saved_rows2.append(result_vector.flatten())
        saved_matrix2 = np.array(saved_rows2)
        saved_matrix2 = gf2_rref(saved_matrix2)
        all_nonzero_rows = saved_matrix2[np.any(saved_matrix2 != 0, axis=1)]
        m3 = all_nonzero_rows.shape[0]
        if m3 > m4:
            saved_rows.append(row)
            saved_rows1.append(result_vector.flatten())

delelogicalZ1 = np.array(saved_rows)
delelogicalZ1 = np.array(saved_rows)
saved_matrix1 = np.array(saved_rows1)
G_12 = delelogicalX2
if delelogicalZ1.shape[0] != 0:
    G_12 = np.vstack((delelogicalX2, delelogicalZ1))

G_12 = gf2_rref(G_12)
G_12=G_12[:16]
m_N = G_12.shape[0]
G3, P1 = create_identity_at_top(G_12)
O_N = np.zeros((m_N, f - m_N))
M = np.hstack((eye(m_N), O_N))
barG1 = np.dot(delelogicalX2, P1.T)
barG1 = np.dot(barG1, M.T) % 2
barD = kernel_gf2(barG1).T
m2 = 0
D = []
# Sparsing matrix D is the same as sparsifying matrix barD, so to simplify that  we make D =barD.
if barD.shape[0] != 0:
    D = barD
    D = np.array(D.tolist())
    m2 = barD.shape[0]
m = filtered_matrix2.shape[0]
m1 = filtered_matrix2.shape[1]
Y1 = m1 * 2 + m2 * 3 + m * 3  # 稀疏化前量子比特开销，3 is the code distance
if m2 != 0:
    h = D.shape[1]
    count_col = [0] * h

    # the qubits cost of Sparsing matrix D and weight(H_G) < = weight(H_X)+2
    for row in D:
        count = 0
        for element in row:
            if element != 0:
                count += 1
        if count > 14:
            log2_count = math.log(count, 13)
            g = math.ceil(log2_count)
            m1 = m1 + 13 * (1 - 13 ** (g - 1)) / (1 - 13)
            m2 = m2 + 13 * (1 - 13 ** (g - 1)) / (1 - 13)
    for j in range(D.shape[1]):
        for i in range(D.shape[0]):
            if D[i][j] != 0:
                count_col[j] += 1
    for j in range(D.shape[1]):
        if count_col[j] > 9:
            log3_count_col = math.log(count_col[j], 8)
            f = math.ceil(log3_count_col)
            m2 = m2 + 8 * (1 - 8 ** (f - 1)) / (1 - 8)
            m1 = m1 + 8 * (1 - 8 ** (f - 1)) / (1 - 8)
Y = m1 * 2 + m2 * 3+ m * 3#稀疏化后量子比特开销
result = (Y1, Y, gg, 0)
print(result)

