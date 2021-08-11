# １週目問題

### Q1
list型であるlist_a, numpy型であるnum_a, Tensor型であるten_aについて、
```
print(list_a)
print(type(list_a))

###
[[2, 3, 4], [1, 3, 3]]  
<class 'list'>
###


print(num_a)
print(type(num_a))


###
[[4. 5. 6. 6.]
 [3. 2. 4. 8.]] 
<class 'numpy.ndarray'>
###


print(ten_a)
print(type(ten_a))
###
tensor([[3., 4., 2., 5., 4.],
        [4., 5., 6., 4., 2.]], dtype=torch.float64)
<class 'torch.Tensor'>
###
```
### Q2
Q1のlist_a, num_a, ten_aについて、それぞれ形状を求めよ。
ちなみに出力結果は以下のようになる。
```
(2, 4)    ##num_aの形状
torch.Size([2, 5])   ##ten_aの形状

```

### Q3
list_b = [[2, 3, 4, 2], [1, 3, 3], [3, 2, 1], [5, 3, 4]]というlistを作成した。出力が
```
[2, 3, 4, 2]
[1, 3, 3]
[3, 2, 1]
[5, 3, 4]
```
となるようなpythonでの処理を考えよ(for文を使う)。

### Q4
list_b = [[2, 3, 4], [7, 8, 9]]について
要素をfloat型にしたnumpy配列であるnum_b, 
num_bをTensor配列に変換したten_bを作成せよ。出力は以下のようになる。
```
print(num_b)
###
[[2. 3. 4.]
 [7. 8. 9.]]
###

print(ten_b)

###
tensor([[2., 3., 4.],
        [7., 8., 9.]], dtype=torch.float64)
###
```
### Q5
num_bを１次元配列に変換したnum_b_1, また、ten_aを1次元配列に変換したten_a_1を作成せよ。出力は以下のようになる。
```
print(num_b_1)
###
[2. 3. 4. 7. 8. 9.]
###

print(ten_b_1)
###
tensor([2., 3., 4., 7., 8., 9.], dtype=torch.float64)
###
```

### Q6
Q4のnum_b,ten_bについて、転置行列num_b_t, ten_b_tを作成せよ。
```
print(num_b_t)
###
[[2. 7.]
 [3. 8.]
 [4. 9.]]
###

print(ten_b_t)
###
tensor([[2., 7.],
        [3., 8.],
        [4., 9.]], dtype=torch.float64)
###
```
