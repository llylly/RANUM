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



