# Abstraction Arithmetic

#### Sub

First align the two tensors via split_by

$[a,b] - [c,d] = [a-d, b-c]$.

#### MatMul

First align the two tensors via split_by

Note that there are multiple cases of matrix multiplications: left dot product, right dot product, normal multiplication with broadcasting.

They are separately handled.

For each case, we need to align the two tensors via split_by. The alignment procedure is slightly different and more involved then Sub, Add, etc.

After the two tensors are splited, for the most normal multiplication, we compute the following abstraction.

$[a,b] \times [c,d] = [\min(ac,ad,bc,bd),\max(ac,ad,bc,bd)].$

#### Add

First align the two tensors via split_by

$[a,b]+[c,d]=[a+c,b+d]$.

#### General Flatten

Suppose the start_dim is $s$, the tensor shape is $[a_1, a_2, ..., a_s, ..., a_d]$.

The split point of abstraction tensor are $\{\mathcal{S}_i\}_{i=1}^d$.

We first check the last dimension that has more than one split from $s$, denoted as $t$.

Formally, $t$ satisfies: ($|\mathcal{S}_t| > 1$ or $t=s$) and $\forall t' > t, |\mathcal{S}_{t'}| = 1$.

Then, we can determine the flattened abstraction tensor's shape: $\left[a_1, a_2, ..., a_{s-1}, \prod_{i=s}^{t-1} a_i \cdot |\mathcal{S}_t|\right]$.

For each element $j \in \{0,1,...,\prod_{i=s}^{t-1} a_i \cdot |\mathcal{S}_t|-1\}$, we need to first compute $\mathrm{ind} = \dfrac{j}{|\mathcal{S}_t|}$, and use $\mathrm{ind}$ to obtain the indexes for $s$ ... ($t-1$) dimension, then map to the abstraction's indexes via $\mathcal{S}_i$, and finally extract the corresponding elements from the abstraction.

#### General Stretch

Suppose the source shape is $[a_1, a_2, ..., a_d]$.

Suppose the target shape is $[b_1, b_2, ..., b_d, ..., b_{d'}]$.

The constraints are $a_i = b_i, 1\le i <d$ and $\prod_{i=d}^{d'} b_i = a_d$.

We first exam all element (i.e., split point) $e$ in $\mathcal{S}_d$ to construct the split points of the target shape.

For each element $e$, we can figure out the corresponding split points in the target shape, and we denote them as $e'=[e'_d, e'_{d+1}, ..., e'_{d_e}]$. Note that $d_e \le d'$.

For example, suppose the source shape is $[4,4]$ and the target shape is $[16]$. For the split point $e=7$ (zero-indexed), the new split points $e'=[1]$ because splitting $7$ and $8$ corresponds to splitting $[0,1]\times [0,3]$ and $[2,3] \times [0,3]$ in the target shape. For the split point $e=14$, the new split points $e'=[2,2]$ because we need to split $2$ from $3$ both on axis 0 and 1 to isolate $0...14$ from $15$.

We let the new split point sets $\mathcal{S}'_i$ to consume the split points of $e'$, i.e., $\mathcal{S}_i' \gets \mathcal{S}_i' \cup \{e'_i\}$.

After the new split points are determined, we construct the new abstraction tensor as a $d$-dimensional tensor, and figure out the index mappings from each cell in the new abstraction tensor to cell of old abstraction tensor.

Then, we use index_select to reads in the data from the old abstraction tensor, and then reshape the new abstraction tensor to the desired $d'$-dimensional tensor.

