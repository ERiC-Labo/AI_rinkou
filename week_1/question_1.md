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
list_b = [[2, 3, 4, 2], [1, 3, 3], [3, 2, 1], [5, 3, 4]]というlistを作成した。出力が
```
[2, 3, 4, 2]
[1, 3, 3]
[3, 2, 1]
[5, 3, 4]
```
となるようなpythonでの処理を考えよ(for文を使う)。

### 