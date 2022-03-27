# 第二天链表（简单）

 ## 剑指35



请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

方法 1 哈希表

简单来说构建字典，key放原链表的node，value放新链表的node

通过dict.get()函数取得key对应的next和random所对应的旧的node并赋值给新链表的node。

![image-20220321161202001](https://cdn.jsdelivr.net/gh/richardzhangy26/Pic/src/image-20220321161202001.png)



![image-20220321161224720](https://cdn.jsdelivr.net/gh/richardzhangy26/Pic/src/image-20220321161224720.png)

提示：dic.get()函数得到字典对应key的value，与dic[key]不同的是若key为空dic.get返回none而dic[key]会报错

```
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
    	dic = {}
    	d
    	
```







# 第8天 动态规划 10-i. 斐波那契数列

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

### 试一试

```
class Solution:
    def fib(self, n: int) -> int:
```



## 剑指offer63.股票的最大利润（121）

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

 

示例 1:

```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

示例 2:

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```



```

```







































































































