#初始状态
#穷举
#转移方程，最值
#最长回文子串
#ababa---bab
'''
通过起始index和长度就能确定
dp[i,j]是回文串=dp[i+1,j-1]是回文串+dp[i]==dp[j]

'''


# def substring(chuang):
#     n=len(chuang)
#     if n<2:
#         return chuang
#
#     dp=[[False]*n for _ in range(n)]
#     max_len=1
#     begin=0
#
#     #base case
#     for i in range(n):
#         dp[i][i]=True
#
#     for L in range(2,n+1):#最大回文子串的长度
#         for i in range(n):#左边界,右边界-左边界+1=L
#             j=L+i-1
#             if j>=n:
#                 break
#
#             if chuang[i]!=chuang[j]:
#                 dp[i][j]=False
#             else:
#                 if j-i<3:
#                     dp[i][j]=True#1,2长度都为回文串
#                 else:
#                     dp[i][j]=dp[i+1][j-1]
#
#             if dp[i][j] and j-i+1>max_len:
#                 max_len=j-i+1
#                 begin=i
#                 # if len(substring(chuang[i+1:j-1]))>max_len:
#                 #     max_len=len(substring(chuang[i+1:j-1]))
#                 #     start_index=i
#
#
#     return chuang[begin:begin+max_len]#chuang[start_index+1:start_index+1+max_len]
# print(substring('ababa'))
#找零钱问题
# base case amount =0,dp[0]=0;amount<0 无解-1
# dp数组含义：dp[n]：amount=n时，至少需要dp[n]的硬币
#dp[11]是怎么一步计算的？ 看硬币列表，dp[11]=min(dp[11-1],dp[11-2],dp[11-5])
# def changecoin(coins,target):
#     def dp(n):
#         if n<0:
#             return -1
#         if n==0:
#             return 0
#         #dp=[666]*(target+1),只需当前的结果
#         #dp[0]=0
#         res=float('INF')
#         for i ,each in enumerate(coins):
#             subproblem=dp(n-each)
#             if subproblem==-1:
#                 continue
#             res=min(subproblem+1,res)
#         if res!=float('INF'):
#             return res
#         else:
#             return  -1
#     return dp(target)
#
#
#     # for i in range(1,target+1):
#     #     temp=666
#     #     for j,each in enumerate(coins):
#     #         if dp[i-each]<temp: dp[i]=min(dp[i],1+dp[i-each]+1)
#     #             temp=dp[i-each]
#     #     dp[i]=temp+1
#     # return dp[target]
#
# print(changecoin([1,2,5],11))
#base case
#明确dp数组的含义
#穷举态，循环
#子问题无解，跳过
#状态转移方程，最值


#最长回文子串
# def  substring(chuan):
#     n=len(chuan)
#     if n<2:
#         return chuan
#     max_len=1
#     begin=0
#     dp=[[False]*n for _ in range(n)]#二维数组记录长度和开始下标
#     for i in range(n):
#         dp[i][i]=True
#     for L in range(2,n+1):
#         for i in range(n):
#             j=L+i-1
#             if j>=n:
#                 break#一定要注意无解的情况。
#
#             if chuan[i]==chuan[j]:
#                 if j-i<3:
#                     dp[i][j]=True
#                 else:
#                     dp[i][j]=dp[i+1][j-1]#核心思想是穷举，因此判断最值时应该是内层循环一个完成，也就是穷举一个例子，才判断。
#
#             else:
#                 dp[i][j]=False
#             if dp[i][j] and j-i+1>max_len:
#                 max_len=j-i+1
#                 begin=i
#     return chuan[begin:begin+max_len]
# print(substring('ababa'))




# class TreeNode:
#     def __init__(self,val=0,left=None,right=None):
#         self.val=val
#         self.left=left
#         self.right=right
# def isSameTree(p:TreeNode,q:TreeNode):
#
#     if p==None and q==None:#因为这里节点是可以为None,不一定是首个根节点，而是每个根节点，只要二者都是None，那就是同一个节点
#         return True
#     elif  p==None or q==None:
#         return False
#     elif p.val!=q.val:
#         return False
#     else:
#         return isSameTree(p.left,q.left)and isSameTree(p.right,q.right)
#
# p=TreeNode(1,TreeNode(2,None,None),None)
# q=TreeNode(1,None,TreeNode(2,None,None))
# print(isSameTree(p,q))
#
class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right
#删除二叉树的某个节点，
#找到了，并删除
'''
1、节点是叶子节点，直接删除
2、节点有一个子节点，就用子节点直接代替，删除原节点
3、节点有两个子节点，因此既可以删除左子树中最大的，也可以删除右子树中最小的，代替原来的
'''
#判断值大小，在左/右子树里面找
#因为有修改，所以返回的是TreeNode,否则直接返回
# def getmin(root:TreeNode):
#     while root.left:
#         root=root.left
#     return root.val
# def deleteNode(root:TreeNode,key:int):
#     if root==None:
#         return None
#     if root.val==key:
#         if root.right==None:
#             return root.left
#         elif root.left==None:
#             return root.right
#         else:
#             minval=getmin(root.right)
#             root.val=minval
#             root.right=deleteNode(root.right,root.val)
#
#
#
#     elif root.val>key:
#         root.left=deleteNode(root.left,key)
#     else:
#         root.right=deleteNode(root.right,key)
#     return root
# p=TreeNode(1,TreeNode(2,None,None),None)
# q=TreeNode(1,None,TreeNode(2,None,None))
# print(isSameTree(p,q))
import sys

# def indexoutofrange(x_1,y_1,n,m):
#     if x_1 in range(n) and y_1 in range(m):
#         return True
#     return False
#
# def road(n, m, x_1, y_1, x_2, y_2, values):
#     new_values=values
#     # new_values = [[-2] * (m + 2) for _ in range(n + 2)]
#     # 外面尾一层-1
#     # for ii in range(n):
#     #     for jj in range(m):
#     #         if values[ii][jj] != 0:
#     #             new_values[ii + 1][jj + 1] = values[ii][jj]
#
#     dp = [[-1] * m for _ in range(n)]  # 当前位置的最少次数
#
#     i = x_1 - 1
#     j = y_1 - 1
#     dp[x_1 - 1][y_1 - 1] = 0
#     # new_i = i + 1
#     # new_y = j + 1
#     while (i != x_2-1 and j != y_2-1):
#
#
#         if indexoutofrange(i-1,j,n,m)and new_values[i- 1][j] > new_values[i][j] or new_values[i- 1][j] ==new_values[i][j]:
#             dp[i][j] = min(dp[i - 1][j]+1,dp[i][j])
#             i=i-1
#
#         if indexoutofrange(i+1,j,n,m)and new_values[i+1][j] > new_values[i][j] or new_values[i+ 1][j] ==new_values[i][j]:
#             dp[i][j] = min(dp[i + 1][j]+1,dp[i][j])
#             i=i+1
#         if indexoutofrange(i,j+1,n,m)and new_values[i][j + 1] > new_values[i][j]or new_values[i][j + 1] == new_values[i][j]:
#             dp[i][j] = min(dp[i][j + 1]+1,dp[i][j])
#             j=j+1
#         if indexoutofrange(i,j-1,n,m)and new_values[i][j - 1] > new_values[i][j] or new_values[i][j - 1] == new_values[i][j]:
#             dp[i][j] = min(dp[i][j - 1]+1,dp[i][j])
#             j=j-1
#         # new_i = i
#         # new_y=j
#
#     return dp[x_2-1][y_2-1]
#
#
# if __name__ == "__main__":
#     #读取第一行的n
#     # linee = sys.stdin.readline().strip()
#     # linee= list(map(int, linee.split()))
#     # n,m=linee[0],linee[1]
#     # linee = sys.stdin.readline().strip()
#     # linee = list(map(int, linee.split()))
#     # x_1, y_1, x_2, y_2 = linee[0],linee[1],linee[2],linee[3]
#     # all_values=[]
#     # for i in range(n):
#     #     # 读取每一行
#     #     line = sys.stdin.readline().strip()
#     #     # 把每一行的数字分隔后转化成int列表
#     #     values = list(map(int, line.split()))  # n*m
#     #     all_values.append(values)
#     n=5
#     m=3
#     x_1=1
#     y_1=1
#     x_2,y_2=5,3
#     all_values=[[4,4,3],[3,5,4],[6,5 ,6],[7,4,10],[8,9,9]]
#     road(n, m, x_1, y_1, x_2, y_2, all_values)
# def  flatten(root:TreeNode):
#     flatten(root.left)
#     flatten(root.right)
#     left=root.left
#
#     right=root.right
#     root.left=None
#     root.right=left
#
#     while root.right:
#         root=root.right
#     root.right=right
# TreeNode6=TreeNode(6,None,None)
# TreeNode5=TreeNode(5,None,TreeNode6)
# TreeNode4=TreeNode(4,None,None)
# TreeNode3=TreeNode(3,None,None)
# TreeNode2=TreeNode(2,TreeNode3,TreeNode4)
# TreeNode1=TreeNode(1,TreeNode2,TreeNode5)
# root = TreeNode1
# flatten(root)
import collections
# trees=collections.defaultdict()#为了防止key不存在,引发异常，其他和dict一样，default_factory初始值为None
# trees.default_factory=trees.__len__#这里因为还没有添加，所以__len__是0，default_factory初始为0
# def findDuplicateSubtrees(root):
#     trees=collections.defaultdict()
#     trees.default_factory=trees.__len__()
#     count=collections.Counter
#     ans=[]
#     def newask(node):
#         if node==None:
#             return '#'
#         left=newask(node.left)#返回字符串
#         right=newask(node.right)
#         uid=','.join(_ for _ in [left,right,str(node.val)])
#         #因为是针对每一个以root为根的树，加入做判断，因此应该这里面，而不是函数外
#         count[uid] += 1
#         if count[uid] == 2:
#             ans.append(node)#在函数里面写函数，就可以调用里面的变量
#         return uid
#     newask(root)
#     return ans
# a=3
# b='4'
# c='5'
# trees[a,b,c]=2
# print(trees[a,b,c])

# class Solution:
#     def inorder(self, node: TreeNode):
#         if node == None:
#             return None
#
#         self.inorder(node.right)
#         self.sumvalue += node.val
#         node.val = self.sumvalue
#         self.inorder(node.left)
#
#     def convertBST(self, root: TreeNode):  # -> TreeNode:
#         # 需要知道大于等于自己的个数，找最右节点的数，
#         # 4:5.sum（26）+4
#         # 中序是升序，改成降序，用额外的sum做累加，赋给节点
#         #不用生成新的TreeNode,这里只是用list形式表示，本身还是树节点，如果用list搞个新返回的，序号应该有问题，最好就是本身上
#         #赋值，只改变val
#         self.sumvalue = 0
#         self.inorder(root)
#         return root
#
# #BST的合法性
# def isvalidnode(root:TreeNode,min:TreeNode,max:TreeNode):
#     if root ==None:
#         return True
#     if min!=None and root.val<min.val or root.val==min.val:
#         return False
#     if max!=None and root.val>max.val or root.val==max.val:
#         return False
#     return isvalidnode(root.left,min,max) and isvalidnode(root.right,min,max)
# def isvalidbst(root:TreeNode):
#    return isvalidnode(root,None,None)
# #BST搜索一个数
# def isinbst(root:TreeNode,target):
#     if root==None:
#         return False
#     if root.val>target:
#         return isinbst(root.left,target)
#     elif root.val<target:
#         return  isinbst(root.right,target)#这里需要返回，才能进行递归，因为整个是判断元素是否存在，是否问题
#     else:
#         return True
#
# def bst(root:TreeNode,target:int):
#     if root.val==target:
#         # root.val=newvalue
#         # newnode=TreeNode(newvalue)
#         # newnode.right=root.right
#         # newnode.left=root.left
#     pass
#     if root.val<target:
#         bst(root.right,target)
#     if root.val>target:
#         bst(root.left,target)
# #一旦涉及修改，返回的TreeNode类型
# #在BST中插入一个数,一般不会插入已经存在的元素
# def insertintobst(root:TreeNode,target:int):
#     if root==None:
#         return TreeNode(target)
#     if root.val>target:
#         root.left= insertintobst(root.left,target)
#     if root.val<target:
#         root.right= insertintobst(root.right,target)
#     return root
# #在BST中删除一个数
# '''
# 1、叶子节点，直接删除
# 2、有一个孩子节点，直接代替，
# 3、有两个孩子节点，就得找右子树最小的/左子树最大的进行替代，再删除
# '''
# #一旦涉及修改，返回的TreeNode类型，对递归调用的返回值进行接收。
# def getminnode(root:TreeNode):
#     p=root
#     while p.right:
#         p=p.right
#     return p
# def deletenode(root:TreeNode,key:int):
#     if root==None:
#         return None
#     if root.val==key:
#         if root.left==None:
#             return root.right
#         if root.right==None:
#             return root.left
#         #创建一个新的TreeNode,替代root就是，改root的值，删除自己
#         minnode=getminnode(root.right)
#         root.val=minnode.val
#
#         root.right=deletenode(root.right,minnode.val)
#
#
#
#     elif root.val<key:
#         root.right=deletenode(root.right,key)
#     elif root.val>key:
#         root.left=deletenode(root.left,key)#y意思就是删除左子树中的key，然后就是一个新的左子树，要的是一个完整的
#         #TreeNode类型的，所以还需要返回root
#     return root

# dp=[[-1]*3 for _ in range(4)]
# print(dp)
#状态、dp含义，base case
#64  最小路径和
# def minPathSum( grid) :
#     if not grid or not grid[0]:
#         return 0
#     m = len(grid)
#     n = len(grid[0])
#     dp = [[0] * n for _ in range(m)]
#     dp[0][0] = grid[0][0]#到该坐标，最小的路径数字总和
#     # for i,each in enumerate (grid):#row
#     #     for j ,ea in enumerate(grid[i]):#col
#     for j in range(1,n):
#         dp[0][j]=dp[0][j-1]+grid[0][j]
#     for i in range(1,m):
#         dp[i][0] = dp[i - 1][0] + grid[i][0]
#
#     for i in range(m):
#         for j in range(n):
#             if i -1 <0 or j -1<0:
#                 continue
#             #这两行应该是base case，这个必须要搞清
#             if dp[i][j]==0:
#                 dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j]# grid[i + 1][j] if dp[i + 1][j] < dp[i][j + 1] else grid[i][j + 1]
#
#     return dp[m - 1][n - 1]
# grid= [[1,2,3],[4,5,6]]#[[1,3,1],[1,5,1],[4,2,1]]
# print(minPathSum(grid))
#11106
#1 1 A
#11 2  AA  K
#111 3  AAA AK KA
# class Solution:
#     def numDecodings(self, s: str) -> int:
#         if s[0] == '0': return 0
#         if len(s) == 1: return 1
#         #这上面考虑的是很短的情况
#         legalstr = set(str(i) for i in range(1, 27))
#         #set查找是O(1)
#         dp = [0] * (len(s))#第i个字符串，有dp[i]种方法
#
#         # base case
#         dp[0] = 1#base case，因为后续有dp[i-1],dp[i-2],所以dp[1]也得知道初始值
#         if s[1] not in legalstr:  # s[1]为0
#             dp[1] = 1 if s[: 2] in legalstr else 0
#         else:
#             dp[1] = 2 if s[: 2] in legalstr else 1
#         # base case
#
#         # 因为要用到i-2 所以至少初始化 dp[0] dp[1]
#         #考虑dp[i]和dp[i-1]...dp[0]的关系
#         for i in range(2, len(s)):
#             if s[i] not in legalstr:
#                 if s[i - 1: i + 1] not in legalstr:
#                     return 0
#                 else:
#                     dp[i] = dp[i - 2]
#             else:
#                 if s[i - 1: i + 1] in legalstr:
#                     dp[i] = dp[i - 1] + dp[i - 2]
#                 else:
#                     dp[i] = dp[i - 1]
#         return dp[-1]

#456<<5<6
# def show(strings):
#     if strings[0]=='<':
#         return None
#     result=[]
#     for i,each in enumerate(strings) :
#         if each =='<' and i-1>0:
#             result.remove(result[-1])
#             continue
#         result.append(each)
#     return ''.join(_ for _ in result)
# print(show('456<<5<6'))
import copy
# result=[]
# def backtrack(nums,track):
#     global result
#     if len(track)==len(nums):
#         temptrack=copy.deepcopy(track)
#         result.append(temptrack)#这里直接把track加进去是有问题的，因为在回溯的过程中，不断地对track进行删除和添加，导致result 里面会不停的变
#         #但其实这里需要加入的是遍历到叶子节点的路径。
#         #copy.deepcopy深度拷贝，完全拷贝了父对象及其子对象，两者是独立的，也就是改变深度拷贝后的和原来的没关系
#         #浅拷贝，a和b式一个独立的对象，但他们的子对象还是指向统一对象
#
#         return
#     for i in range(len(nums)):
#         #这个是判断是否已经添加到路径列表中，因为不存在重复的元素
#         if nums[i] in track:
#             continue
#
#         track.append(nums[i])
#         backtrack(nums,track)
#         #上面是已经遍历完一个从根到叶子节点的过程，回溯的话，需要将路径清除，撤销过程
#         track.remove(track[-1])
#
# def permute(nums):
#     track=[]
#     backtrack(nums,track)
#     return result
# print(permute([1,2,3]))
# result = []
# def isvalid( board, row, col):
#
#     for j in range(len(board)):
#         if board[j][col] == 'Q':  # 看这一列的上面有没有和她冲突地
#             return True
#     # 左上有没有冲突[row-1]，是左上这一个斜线上都不能有
#     for i in range(row - 1, -1, -1):  # j in range(row-1,len(board)):
#         for j in range(col - 1, -1, -1):
#
#             if board[i][j] == 'Q':
#                 return True
#     # 右上有没有冲突，是右上这一个斜线上都不能有
#     for i in range(row - 1, -1, -1):
#         for j in range(col + 1, len(board)):
#
#             if board[i][j] == 'Q':
#                 return True
#     return False
#
# def totalNQueens( n: int) -> int:
#     global result
#     board = [['.'] * n] * n
#
#     def backtrack(board, row):
#         if row == len(board):
#             temp=copy.deepcopy(board)
#             result.append(temp)
#             return
#         for i in range(n):
#             if isvalid(board, row, i):
#                 continue
#             board[row][i] = 'Q'
#             backtrack(board, row + 1)
#             board[row][i] = '.'
#
#     backtrack(board, 0)
#     return len(result)
# print(totalNQueens(4))
# def reverse( x: int) -> int:
#     # 把x当成字符串，逆序输出，再转成int
#     # 长度为1，就是本身；超过范围，返回0
#     if x <= -2 ** 31 or x >= (2 ** 31) - 1:
#         return 0
#     strings = str(x)
#     if len(strings) == 1:
#         return x
#     new_strings = ''.join(strings[i] for i in range(len(strings)-1, -1, -1))
#     return int(new_strings)
# reverse(123)
# def letterCombinations( digits: str) :
#     # 全排列
#     if len(digits) == 0:  # index < len(digits) and digits[index] == '':
#         return []
#     digdict = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], '4': ['g', 'h', 'i'], '5': ['j', 'k', 'l'],
#                '6': ['m', 'n', 'o'], '7': ['p', 'q', 'r', 's'], '8': ['t', 'u', 'v'], '9': ['w', 'x', 'y', 'z']}
#     result = []
#     track = []
#
#     def backtrack(digits, index):
#
#         # if digits[index]==1:
#         #     return digdict[digits]
#         if len(track) == len(digits):
#             temp = copy.deepcopy(track)
#             temp = ''.join(_ for _ in temp)
#             result.append(temp)
#             return
#
#         for i in digdict[digits[index]]:
#
#             track.append(i)
#             backtrack(digits, index + 1)
#             track.remove(track[-1])
#
#     backtrack(digits, 0)
#     return result
# print(letterCombinations('777'))
#["www","wwx","wwy","wwz","wxw","xwx","wxy","wxz","wyw","ywx","ywy","wyz","wzw","zwx","zwy","zwz","xww","xwx","wxy","wxz","wxw","xwx","wxy","wxz","wyw","ywx","ywy","wyz","wzw","zwx","zwy","zwz","yww","ywx","ywy","wyz","wxw","xwx","wxy","wxz","wyw","ywx","ywy","wyz","wzw","zwx","zwy","zwz","zww","zwx","zwy","zwz","wxw","xwx","wxy","wxz","wyw","ywx","ywy","wyz","wzw","zwx","zwy","zwz"]
#["www","wwx","wwy","wwz","wxw","wxx","wxy","wxz","wyw","wyx","wyy","wyz","wzw","wzx","wzy","wzz","xww","xwx","xwy","xwz","xxw","xxx","xxy","xxz","xyw","xyx","xyy","xyz","xzw","xzx","xzy","xzz","yww","ywx","ywy","ywz","yxw","yxx","yxy","yxz","yyw","yyx","yyy","yyz","yzw","yzx","yzy","yzz","zww","zwx","zwy","zwz","zxw","zxx","zxy","zxz","zyw","zyx","zyy","zyz","zzw","zzx","zzy","zzz"]
# def threeSum( nums) :
#     if not nums or len(nums) < 3:
#         return []
#     zero_count = 0
#     for _ in nums:
#         if _ == 0:
#             zero_count += 1
#
#     if zero_count == len(nums):
#         return [[0, 0, 0]]
#     result = []
#     track = []
#
#     def backtrack(nums):
#         if len(track) == 3:
#             if sum(track) == 0:
#                 temp = sorted(copy.deepcopy(track))
#                 if temp not in result:
#                     result.append(temp)
#             return
#             # if temp not in result:
#             #     result.append(temp)
#
#         for i, ea in enumerate(nums):
#             # if i in track:
#             #     continue
#             track.append(ea)
#             # newnums=nums.remove(nums[i])
#             backtrack(nums)
#             track.remove(track[-1])
#
#     backtrack(nums)
#     if [0, 0, 0] in result:
#         result.remove([0, 0, 0])
# def threeSum( nums) :
#     if not nums or len(nums) < 3:#特例判断，为空或者不够3个肯定不行
#         return []
#     nums = sorted(nums)#排序后，对于开始就大于0肯定不行
#     result = []
#
#     for i in range(len(nums)):#双重指针+去重，
#         if nums[i] > 0:
#             continue
#         if i>0 and nums[i]==nums[i-1]:
#             continue#因为不计算重复的3元组
#         left = i+1
#         right = len(nums) - 1
#         while (left < right ):
#
#
#             if nums[i] + nums[left] + nums[right] == 0:
#                 result.append([nums[i], nums[left], nums[right]])
#                 while(left<right and nums[left]==nums[left+1]):#不计算重复的3元组
#                     left=left+1
#                 while(left<right and nums[right]==nums[right-1]):#不计算重复的3元组
#                     right=right-1
#                 left=left+1
#                 right=right-1
#             elif nums[i] + nums[left] + nums[right] < 0:
#                 left=left+1
#             elif nums[i] + nums[left] + nums[right] > 0:
#                 right=right-1
#
#     return result
# print(threeSum([-1,0,1,2,-1,-4]))
# def combinationSum(candidates, target):
#     # 结束条件就是sum==target
#     # 选择列每次都是candidates
#     result = []
#     track = []
#
#     def backtrack(candidates, track):
#         if sum(track) == target:
#             temp=sorted(track)
#             if temp not in result:
#                 result.append(temp)
#             return
#
#         for i, ea in enumerate(candidates):
#             if sum(track)+ea > target :
#                 continue
#             track.append(ea)
#             backtrack(candidates, track)
#             track.pop()#track.remove(track[-1])
#
#     backtrack(candidates, track)
#
#     return result
# def combinationSum(candidates, target):
#     # 结束条件就是sum==target
#     # 选择列每次都是candidates
#     result = []
#     track = []
#
#     def backtrack(sumnow,startindex):
#         if sumnow == target:
#             result.append(track[:])
#             return
#
#         for i in range(startindex,len(candidates)):
#             sumnow+=candidates[i]
#             if sumnow> target :
#                 sumnow-=candidates[i]
#                 continue
#             track.append(candidates[i])
#             backtrack(sumnow,startindex)
#             sumnow-=candidates[i]
#             track.pop()#track.remove(track[-1])
#
#     backtrack(0,0)
#
#     return result
# print(combinationSum([7,3,2],18))
# #[[2,2,2,2,2,2,2,2,2],[2,2,2,2,2,2,3,3],[2,2,2,2,3,7],[2,2,2,3,3,3,3],[2,2,7,7],[2,3,3,3,7],[3,3,3,3,3,3]]
# def minDepth(self, root: TreeNode) -> int:
#     if not root:
#         return 0
#     track = []
#     depth = 1
#
#     def backtrack(root, depth):#dfs
#         if not root.right and not root.left:#满足条件，加进路径中，照的是最小长度的叶子节点
#             track.append(depth)
#             return
#         depth += 1#做选择
#         if root.left:#在选择列表中继续做选择
#             backtrack(root.left, depth)
#         if root.right:
#             backtrack(root.right, depth)
#         depth -= 1#撤销选择
#
#     backtrack(root, depth)
#     return min(track)

    # q=collections.deque([root])#bfs 用的队列
    # depth=1
    # while(q):队列不为空时
    #     sz=len(q)每次都判断队列中的元素长度
    #     for i in range(sz):从sz中开始扩散
    #         cur=q.popleft()弹出这个节点
    #         if not cur.left and not cur.right:#是否满足条件
    #             return depth
    #         if cur.left:#把该节点的左右加进来
    #             q.append(cur.left)
    #         if cur.right:
    #             q.append(cur.right)
    #     depth+=1#完成一层
    # return depth
# def plusone( cur, j):
#     if cur[j] == '9':
#         cur=cur[:j] + '0' + cur[j + 1:]
#     else:
#         temp = str(int(cur[j]) + 1)
#         cur =cur[:j] + temp + cur[j + 1:]
#     return cur
#
#
# def minusone( cur, j):
#     if cur[j] == '0':
#         cur = cur[:j] + '9' + cur[j + 1:]
#     else:
#         temp = str(int(cur[j]) - 1)
#         cur=cur[:j] + temp + cur[j + 1:]
#     return cur
#
# def openLock( deadends, target) :
#     if target in deadends:
#         return -1
#     que = collections.deque(['0000'])#bfs固定需要队列
#     visited = set('0000')
#     # for _ in deadends:
#     #     visited.add(_)
#     onecount = 0#记录旋转次数
#     while que:#队列不为空，
#         sz = len(que)#队列实际大小固定
#         for i in range(sz):#当前队列中的所有节点向四周扩散
#             cur = que.popleft()#
#             if cur in deadends:
#                 continue
#             if cur == target:#判断是否达到条件
#                 return onecount
#             for j in range(4):#将相邻节点加入队列中
#                 up = plusone(cur, j)
#                 if up not in visited:
#                     que.append(up)
#                     visited.add(up)
#                 down = minusone(cur, j)
#                 if down not in visited:
#                     que.append(down)
#                     visited.add(down)
#         onecount += 1
#     return -1
# print(openLock(["0201","0101","0102","1212","2002"],"0202"))
# a='1523'
# b=(3,4,2)
# b[0]=4# 'tuple' ,'str'object does not support item assignment,因为是不可变序列
# #list,set,dict是可变序列
#dfs岛屿数量
# def backtrack(grid,x,y):
#     grid[x][y]='2'
#     for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
#         if 0<=i<len(grid) and 0<=j<len(grid[0]):
#             backtrack(grid,x,y)
# def numisland(grid):
#     nr=len(grid)
#     if nr==0:
#         return 0
#     nc=len(grid[0])
#     numland=0
#     for i in range(nr):
#         for j in range(nc):
#             if grid[i][j]=='1':
#                 numland+=1
#                 backtrack(grid,i,j)
#                 #做选择
# #bfs
# def quicksort(nums,start,end):
#     if start>=end:
#         return
#     pivot=nums[start]
#     low=start#start和end是固定的，low和high 是每次移动的
#     high=end
#     while low<high:
#         while low<high and nums[high]>=pivot:
#             high-=1
#         nums[low]=nums[high]
#         while low<high and nums[low]<pivot:
#             low+=1
#         nums[high]=nums[low]
#     nums[low]=pivot
#     quicksort(nums,start,low-1)
#     quicksort(nums,low+1,end)
#
# def merge(left,right):
#     l,r=0,0
#     result=[]
#     while l<len(left) and r<len(right):
#         if left[l]<=right[r]:#比较头部，小的加进新和成的result中
#             result.append(left[l])
#             l+=1
#         else:
#             result.append(right[r])
#             r+=1
#     result+=left[l:]#这里说明可能一个列表为空，就把剩下的两个列表直接加在result后面，因为本身这俩就是有序列表
#     result+=right[r:]
#     return result
#
# def merge_sort(nums):
#     if len(nums)<=1:
#         return nums#所以是要返回的
#     mid=len(nums)//2
#     left=merge_sort(nums[:mid])#这一步是在不停地划分，一直划分到每组只有一个元素
#     right=merge_sort(nums[mid:])
#     return merge(left,right)#从每组只有一个元素开始进行两两合并，合并后的组是有序的
# nums=[54, 26, 93, 17, 77, 31, 44, 55, 20]
#
# def select_sort(nums):
#     n=len(nums)
#     for i in range(n-1):#这里应该是n-1,因为后面的j是从i+1开始的，所以是n的话，就是n+1,那就有问题
#         min_index=i
#         for j in range(i+1,n):
#             if nums[j]<nums[min_index]:
#                 min_index=j
#         if min_index!=i:
#             nums[min_index],nums[i]=nums[i],nums[min_index]
#     return nums#这个排升序
# def select_sort_2(nums):
#     n=len(nums)
#     for i in range(n-1):
#         max_index=i
#         for j in range(i+1,n):
#             if nums[j]>nums[max_index]:
#                 max_index=j
#         if max_index!=i:
#             nums[max_index],nums[i]=nums[i],nums[max_index]
#     return nums#这个是排降序
# #quicksort(nums,0,len(nums)-1)
# #print(merge_sort(nums))
# print(select_sort_2(nums))

# def generateParenthesis(n):
#     #枚举括号生成方式
#     #判断合理性
#     #dfs,枚举加进路径，选择列表每次都是‘（’或‘）’
def isyouxiao( track):
    # 搞一个栈，碰见右括号，就弹出-1，栈为空就是有效的
    tempzhan = []
    for i in track:
        if i == '(':
            tempzhan.append(i)
        if i == ')':
            tempzhan.remove(tempzhan[-1])
    if len(tempzhan) == 0:
        return True
    else:
        return False

#给定括号对数，穷举有效括号方式
# def generateParenthesis( n) :
#     # 枚举括号生成方式
#     # 判断合理性
#     # dfs,枚举加进路径，选择列表每次都是‘（’或‘）’
#     result = []
#     track = []  # (())  ()()  ()
#     # if n == 1:
#     #     return ['()']
#     def backtrack(track,rn,ln):
#         if len(track)==n*2:
#             #if isyouxiao(track):
#             result.append(''.join(track))
#             return
#         if ln<n:#因为这是在有效的加左右括号，所以不用判断有效性
#             track.append('(')
#             ln+=1#上面这俩个就是做选择
#             backtrack(track,rn,ln)
#             track.pop()
#             ln-=1#这个才是撤销选择
#         if rn<ln:#这个不是rn<n,因为是从（开始，括号成对的
#             track.append(')')
#             rn+=1
#             backtrack(track,rn,ln)
#             track.pop()
#             rn-=1
#
#
#     backtrack(track,0,0)
#     return result
#
# def generateParenthesis(n):
#         def generate(A):
#             if len(A)==2*n:
#                 if valid(A):
#                     ans.append("".join(A))
#                     return
#             #else:#这里加else的原因是对于长度大于2n，不用else，就会一直加，加了else，长度超过了就pop,要么在循环里跳过，要么if else
#             for i in ['(', ')']:
#                 if len(A)>2*n:
#                     continue
#                 A.append(i)
#                 print('a',A)
#                 generate(A)
#                 A.pop()
#                 print('b',A)
#             #     A.append('(')
#             #     generate(A)
#             #     A.pop()
#             #     A.append(')')
#             #     generate(A)
#             #     A.pop()
#
#         def valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(': bal += 1
#                 else: bal -= 1
#                 if bal < 0: return False
#             return bal == 0
#
#         ans = []
#         generate([])
#         return ans
# print(generateParenthesis(2))
#从左上到右下有多少条不同的路径
# def uniquePaths( m, n):
#     # 动态规划
#     # base case 第一行和第一列都是直接初始化得到，因为只能向右或向下移动一步
#     # df的含义，到达当前坐标有多少条不同的路径,自底向上
#     # 状态转移方程，i和i-1....1之间的关系，x,y-1  x-1,y df[x][y]=df[x][y-1]+df[x-1][y]
#     df = [[-1]*n  for _ in range(m)]#[[-1] * n] * m第一种方式是初始化m次，每次都是[-1]*n ,而[[-1] * n] * m是[-1] * n]重复m次，导致对某一行操作，所有行都操作
#     df[0][0] = 1  # 只有一种原地不动，不能是0，df意味着到达该坐标可以走的路径
#     #[[1]*n]+[[1]+[0]*(n-1) for _ in range(m-1)]前面的【1】*n得在二维数组里，所以是[[1]*n]
#     for i in range(1, n):  # 列
#         df[0][i] = 1  # df[0][i-1]+1
#     for i in range(1, m):  # 行
#         df[i][0] = 1  # df[i-1][0]+1
#     print(df)
#     for j in range(1, m):
#         for k in range(1, n):
#             print('a',df[j][k])
#             if df[j][k] == -1:##其实这里有没有这个判断，是无所谓的，因为除去第一行和第一列本身就是-1
#                 df[j][k] = df[j][k - 1] + df[j - 1][k]
#             print('b', df[j][k])
#     return df[m - 1][n - 1]
#
# print(uniquePaths(3,7))
import logging
# def use_logging(func):
#     logging.warning("%s is running"%func.__name__)
#     func()
# def bar():
#     print('i am bar')
# use_logging(bar)#破坏原有的代码逻辑结构，还是用bar()的方式
#装饰器
# def use_logging(func):#装饰器，把func wrapper
#     def wrapper(*args,**kwargs):
#         logging.warning('%s is running'%func.__name__)
#         return func(*args,**kwargs)#返回函数对象
#     return wrapper
# @use_logging#用这个定义函数，避免bar=use_logging(bar)赋值操作
# def bar():
#     print('i am bar')
# @use_logging
# def foo():
#     print('i am foo')
# bar()
# foo()
#带参的装饰器 level
# def use_logging(level):#对原有装饰器进行函数封装，返回一个装饰器
#     def decorator(func):#
#         def wrapper(*args,**kwargs):
#             if level=='warn':
#                 logging.warning('%s is running'%func.__name__)
#             return func(*args)
#         return wrapper
#     return decorator
# @use_logging(level='warn')
# def foo(name='foo'):
#     print('i am %s'%name)
# foo()
#类装饰器
# class Foo(object):
#     def __init__(self,func):
#         self._func=func
#     def __call__(self):
#         print('class decorator runing')
#         self._func()
#         print('class decorator ending')
# @Foo#类装饰器附加到函数bar上，就会调用类内部__call__方法
# def bar():
#     print('bar')
# bar()
#函数装饰器缺点是原函数的原信息不见了，
# def logged(func):
#     def with_logging(*args,**kwargs):
#         print (func.__name__+'was called')
#         return func(*args,**kwargs)
#     return with_logging
# @logged
# def f(x):
#     '''does some match'''
#     return x+x*x
# print(f.__name__)
# print(f.__doc__)
# '''with_logging
# None
# '''
# f(3)#本来func.__name__应该是要传的f函数的名字，原函数的原信息不见了，
#利用functools.wraps,wraps本身也是一个装饰器，可以把原函数
#的原信息拷贝到装饰器函数中。
# from functools import wraps
# def logged(func):
#     @wraps(func)#原函数的原信息拷贝到装饰器函数中
#     def with_logging(*args,**kwargs):
#         print (func.__name__+' was called')
#         return func(*args,**kwargs)
#     return with_logging
# @logged
# def f(x):
#     '''does some match'''
#     return x+x*x
# print(f.__name__)
# print(f.__doc__)
# '''f
# does some match'''
# from multiprocessing import Process,Queue
# import requests
# import time
# def crawl_process(queue,i):
#     while not queue.empty():
#         try:
#             url=queue.get()
#             r=requests.get(url,timeout=3)
#             print('i am %dth process'.format(i),url,r.status_code)
#         except Exception as e:
#             print(e)
#         return
#
# if __name__ == '__main__':
#     queue = Queue()
#     urls = []#G:\study\CloudMusicSimilarMan-master\urls
#     with open("G:/study/CloudMusicSimilarMan-master/urls.txt") as fp:
#         for url in fp:
#             urls.append(url.strip())
#     print("一共%s个url" %len(urls))
#     for url in urls:
#         queue.put(url)
#
#     start = time.time()
#     print("********************** 开始计时 **********************")
#     p_list = []#
#     for i in range(1,5):
#         p = Process(target=crawl_process, args=(queue,i)) #多进程,创建一个process对象，然后调用它的start()方法来生成进程，target传的函数，args是函数的参数
#         p_list.append(p)
#         p.start()
#         print(p)
#     for p in p_list:
#         p.join()#p.start()后调用p.join()主程序main会等待子程序p运行完，再继续往下走
#         print(p)
#     end = time.time()
#     print("********************** 结束计时 **********************")
#     print("总耗时：",end - start)
#二分查找左右边界
# def leftbounder(nums,target):
#     left=0
#     right=len(nums)-1
#     #闭区间
#     while(left<=right):
#         mid=left+(right-left)//2
#         if nums[mid]>target:
#             right=mid-1
#         elif nums[mid]<target:
#             left=mid+1
#         elif nums[mid]==target:
#             right=mid-1
#     #循环结束，left==right+1
#     #nums在左侧边界里的意思是，有几个元素小于target
#     if left>=len(nums) or nums[left]!=target:#-1是不存在左侧边界,这个是因为会越界，left=right+1
#         return -1
#     return left
# def halfleftbounder(nums,target):
#     left=0
#     right=len(nums)
#     #左开右闭区间
#     while left<right:#还差一个元素没判断
#         mid=left+(right-left)//2
#         if nums[mid]>target:
#             right=mid#[left,mid)
#         elif nums[mid]<target:
#             left=mid+1#[mid+1，right)
#         elif nums[mid]==target:
#             right=mid
#     #循环结束，left==right
#     ##nums在左侧边界里的意思是，有几个元素小于target
#     if left>=len(nums)or nums[left]!=target:#说明不存在比它小的#nums[mid]<target:left=mid+1
#         return -1
#     return left
#
# print(leftbounder([1,2,2,4,5],8))
# print(halfleftbounder([1,2,2,4,5],8))
# def rightbounder(nums,target):
#     if len(nums)==0:
#         return -1
#     left=0
#     right=len(nums)-1
#     while left<=right:
#         #全必区间
#         mid=left+(right-left)//2
#         if nums[mid]>target:
#             right=mid-1
#         elif nums[mid]<target:
#             left=mid+1
#         elif nums[mid]==target:
#             #右侧边界,缩小下界
#             left=mid+1
#     #-1的情况，不需要补left=right+1#因为是右侧边界，当target比所有元素都小，符合nums[mid]>target,right=mid-1会越界
#     if right<0 or nums[right]!=target :return -1#if left==0 or nums[left-1]!=target: return -1
#     return right#
# def halfrightbounder(nums,target):
#     if len(nums)==0:
#         return -1
#     left=0
#     right=len(nums)#左闭右开
#     while left<right:
#         mid=left+(right-left)//2
#         if nums[mid]>target:
#             right=mid
#         elif nums[mid]<target:
#             left=mid+1
#         else:
#             left=mid+1
#     #循环结束，left=right
#     #-1的情况
#     if left==0 or nums[left-1]!=target:#因为右侧边界全在左边nums[mid]>target
#         return -1
#     return left-1
#
# print(leftbounder([2,2,2,4,5],1))
# print(halfleftbounder([2,2,2,4,5],1))

#两数相除，整除
def divide( dividend, divisor):
    f1, f2 = 1, 1
    if divisor < 0:
        divisor = -divisor
        f1 = -1
    if dividend < 0:
        dividend = -dividend
        f2 = -1
    #搞上符号位f1,f2，全部搞成绝对值

    if divisor > dividend:
        return 0
    #这个是针对除出来是0.xx的情况
    lists = [divisor]
    count = [1]
    ans = 0
    while lists[-1] < dividend:
        lists.append(lists[-1] + lists[-1])#这个是倍加
        count.append(count[-1] + count[-1])#这个是商
    #循环结束，list[-1]>=dividend
    while dividend >= divisor:
        if lists[-1] > dividend:
            lists.pop()
            count.pop()
        else:
            dividend -= lists[-1]
            ans += count[-1]
            lists.pop()
            count.pop()

    if ans * f1 * f2 > 2147483647:
        return 2147483647
    elif ans * f1 * f2 < -2147483648:
        return -2147483648
    else:
        return ans * f1 * f2
#print(divide(10,3))
#仅使用多线程
# from threading import Thread
# import queue
# import requests,time
# def crawl_process(queue,i):
#     while not queue.empty():
#         try:
#             url=queue.get()
#             r=requests.get(url,timeout=3)
#             print('我是第{}个线程'.format(i),url,r.status_code)
#         except Exception as e:
#             print(e)
#     return
# if __name__=='__main__':
#     queue=queue.Queue()
#     urls=[]
#     with open('G:/study/CloudMusicSimilarMan-master/urls.txt')as fp:
#         for url in fp:
#             urls.append(url.strip())
#     print("一共{}个url".format (len(urls)))
#     for url in urls:
#         queue.put(url)
#     start=time.time()
#     print("********************** 开始计时 **********************")
#     t_list=[]
#     for i in range(1,5):
#         t=Thread(target=crawl_process,args=(queue,i))#创建thread对象，每个thread对象代表一个线程
#         t_list.append(t)
#         t.start()#线程启动
#         print(t)
#     for t in t_list:
#         t.join()#join()会一直等待对应线程的结束，但可以通过参数赋值，等待规定的时间
#         print(t)
#     end=time.time()
#     print("********************** 结束计时 **********************")
#     print("总耗时：", end - start)
#仅使用协程
# from gevent import monkey
# import  gevent
# monkey.patch_all()
# from tornado.queues import Queue
# import requests,time
#
# def crawl(urls,i):
#     while urls:
#         url = urls.pop()
#         try:
#             r = requests.get(url,timeout = 3)
#             print("我是第{}个【协程】" .format(i),url,r.status_code)
#         except Exception as e:
#             print(e)
# def crawl_gevent(queue):
#     url_list=[]
#     tasks=[]
#     i=0
#     while not queue.empty():
#         url=queue.get()._result
#
#         url_list.append(url)
#         if len(url_list)>1:
#             i+=1
#             tasks.append(gevent.spawn(crawl,url_list,i))#创建协程并启动
#             url_list=[]
#     gevent.joinall(tasks)
# if __name__ == '__main__':
#     queue = Queue()
#     urls = []
#     with open("G:/study/CloudMusicSimilarMan-master/urls.txt") as fp:
#         for url in fp:
#             urls.append(url.strip())
#     print("一共%s个url" % len(urls))
#     for url in urls:
#         queue.put(url)
#
#     start = time.time()
#     print("********************** 开始计时 **********************")
#     crawl_gevent(queue)
#     end = time.time()
#     print("********************** 结束计时 **********************")
#     print("总耗时：",end - start)
# from gevent import monkey
# import gevent
# monkey.patch_all()
# from multiprocessing import Process
# from tornado.queues import  Queue
# import requests
# import time
# def crawl(url,i):
#     try:
#         r = requests.get(url,timeout = 3)
#         print("我是第%s个【进程+协程】" %i,url,r.status_code)
#     except Exception as e:
#         print(e)
# def task_gevent(queue,i):
#     url_list = []
#     while not queue.empty():
#         url = queue.get()._result
#         url_list.append(url)
#         if len(url_list) == 6:
#             tasks = []
#             for url in url_list:
#                 tasks.append(gevent.spawn(crawl,url,i))
#             gevent.joinall(tasks)
#     return
# if __name__ == '__main__':
#     queue = Queue()
#     urls = []
#     with open("G:/study/CloudMusicSimilarMan-master/urls.txt") as fp:
#         for url in fp:
#             urls.append(url.strip())
#     print("一共%s个url" % len(urls))
#     for url in urls:
#         queue.put(url)
#
#     start = time.time()
#     print("********************** 开始计时 **********************")
#     p_list=[]
#     for i in range(1,5):
#         p=Process(target=task_gevent,args=(queue,i))
#         p.start()
#         p_list.append(p)
#         print(p)
#     for p in p_list:
#         p.join()
#         print(p)
#     end = time.time()
#     print("********************** 结束计时 **********************")
#     print("总耗时：", end - start)
#编辑距离
def minDistance(word1,word2):
    m,n=len(word1),len(word2)
    if n*m==0:
        return n+m
    dp=[[0]*(n+1)for _ in range(m+1)]
    #因为是倒着遍历的，所以肯定有越界i-1,j-1
    for i in range(1,m+1):
        dp[i][0]=i
    for j in range(1,n+1):
        dp[0][j]=j
    #上面是base case
    for i in range(1,m+1):
        for j in range(1,n+1):
            if word1[i-1]==word2[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1
    return dp[m][n]
#print(minDistance('horse','ros'))

#链表里是否有环
def hasCycle(self, head):
    # write code here
    # 快慢指针，快慢指针相遇表明环存在
    if not head:
        return False
    slow, fast = head, head
    while fast != None and fast.next != None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def hasCycle2(self, head):
    # write code here
    # 快慢指针，快慢指针相遇表明环存在
    # 有环的话，对于set一定会重复
    if not head:
        return False
    setnode = set()
    while head:
        if head in setnode:
            return True
        setnode.add(head)
        head = head.next
    return False

def hasCycle3(self, head):
    # write code here
    # 快慢指针，快慢指针相遇表明环存在
    # 有环的话，对于set一定会重复
    # 删除节点，节点的下个节点是自己，表明有环存在
    if not head or not head.next:
        return False
    if head.next == head:
        return True
    nextnode = head.next
    head.next = head
    return self.hasCycle(nextnode)
#19. 删除链表的倒数第 N 个结点
# def removeNthFromEnd(self, head, n):
#     # 遍历一次知道长度5,5-n就是正着数的从0开始
#     # length=1
#     # p=head.next
#     # while p:
#     #     length+=1
#     #     p=p.next
#     # dummy=ListNode(0,head)#对于头结点的判断，搞一个亚节点作为头结点的前驱，使得删除操作统一
#     # q=dummy
#     # for i in range(1,length-n+1):
#     #     q=q.next
#     # q.next=q.next.next
#     # return dummy.next
#     # 栈，先进后出，弹出倒数n个节点，到它的前驱，然后连接后继节点
#     dummy = ListNode(0, head)
#     stack = list()
#     cur = dummy
#     while cur:
#         stack.append(cur)
#         cur = cur.next
#     for i in range(n):
#          stack.pop()  # 这里存的是节点啊啊
#     prev = stack[-1]  # 到它的前驱
#     prev.next = prev.next.next
#     return dummy.next
#链表中的节点每k个一组翻转
def reverseN( a, b):
    pre, cur, nex = None, a, a
    while cur != b:  # 这里都已经把a赋值给cur,nex了a!=b:
        nex = cur.next
        cur.next = pre
        pre = cur
        cur = nex
    return pre  # cur需要返回的是翻转后的头结点
def reverseKGroup( head, k):
    # 迭代翻转，[a,b)先翻转k个，之后从下一个位置到末尾再迭代翻转k个，子问题
    # base case是小于k的话，就不用翻转
    # 翻转[a,NULL)的链表和[a,b)类似
    if not head:
        return None
    a, b = head, head
    for i in range(k):
        if not b:  # not b.next:，不满足k的情况下，出现节点为空，说明已经到链表尾结点，所以返回Head
            return head
        b = b.next
    newlisthead = reverseN(a, b)
    a.next = reverseKGroup(b, k)
    return newlisthead

#二叉树的三种遍历方式
class Solution:
    def preorder(self, result, root):
        if root != None:  # 如果正面不好想，比如不知道这里root为空时返回时，就考虑反面
            result.append(root.val)
            self.preorder(result, root.left)
            self.preorder(result, root.right)
    def inorder(self, result, root):
        if root != None:
            self.inorder(result, root.left)
            result.append(root.val)
            self.inorder(result, root.right)
    def postorder(self, result, root):
        if root != None:
            self.postorder(result, root.left)
            self.postorder(result, root.right)
            result.append(root.val)
    def threeOrders(self, root):
        # write code here
        pre, inn, post = [], [], []
        self.preorder(pre, root)
        self.inorder(inn, root)
        self.postorder(post, root)
        return [pre, inn, post]
class Solution1:
    #二叉树的三种遍历方式
    def threeOrders(self , root ):
        # write code here
        pre,inn,post=[],[],[]
        def find(root):
            if not root:
                return None
            pre.append(root.val)
            find(root.left)
            inn.append(root.val)
            find(root.right)
            post.append(root.val)
        find(root)
        return [pre,inn,post]
#两数之和等于target
def twoSum( numbers, target):
    # write code here
    # dfs
    #         result=[]
    #         num2index=dict()
    #         for i,ea in enumerate(numbers):
    #             num2index[ea]=i
    #         for j,ea in enumerate(numbers):
    #             if target-ea not in numbers[j+1:]:
    #                 continue
    #             else:
    #                 result.append(j+1)
    #                 result.append(num2index[target-ea]+1)
    #                 break
    #         return result
    #dfs的方法可以节，但是针对这种只有一种唯一解，最好是在找到就返回，dfs必须要试完所有节点，耗时
    result = []
    track = []

    def backtrack(numbers, track):
        if len(track) == 2 and track[0] < track[1]:
            if numbers[track[0]] + numbers[track[1]] == target:
                result.append(track[:])
            return
        #             elif len(track)>2:
        #                 return

        for i in range(len(numbers)):
            if len(track) > 2 or i in track:  # 这里还有一个条件，就是不走回头路
                continue
            if len(track) > 0 and i < min(track):
                continue

            track.append(i)
            backtrack(numbers, track)
            track.pop()

    backtrack(numbers, track)
    result1 = [_ + 1 for _ in result[0]]
    return result1

def eachcost( ea, mid):  # 这个是计算以mid为speed判断piles中每一个需要耗费几个1小时，
    # piles[i]-=i if i<mid else mid
    # return i if i<mid  else mid
    #比如speed=6,对于piles[1]=6需要耗费1个小时，对于piles[2]=7需要耗费2个小时，1个小时吃6根，另1个小时吃1根
    return (ea // mid) + (1 if ea % mid > 0 else 0)

def canfinish( piles, h, mid):
    # 吃不吃得完主要判断时间<=H
    time = 0
    for i, ea in enumerate(piles):#计算以mid速度吃完所有香蕉需要的时间，小于h中的最小速度
        time += eachcost(ea, mid)
    return time <= h
#koko吃香蕉的最小速度
def minEatingSpeed( piles, h):
        # 暴力解法
        # speed最小1，最大是max(piles),在这个线性空间进行索引，查找一个吃完speed的时间<=H
        # initspped=1
        # while initspped<max(piles):
        #     if canfinish():
        #         return initspped
        #     initspped+=1
        # 因为求的是最小速度 ，因此相当于二分查找的左侧边界

    left = 1
    right = max(piles) + 1
    while left < right:
        # 左开右闭的区间
        mid = left + (right - left) // 2
        if canfinish(piles, h, mid):
            right = mid
        else:
            left = mid + 1
    return left
#print(minEatingSpeed([3,6,7,11],8))

#最长无重复子数组
class Solution_no_repeat_sub_array:
    def norepeat(self,temparr):
        setarr=set(temparr)
        if len(setarr)==len(temparr):
            return True
        else:
            return False
    def maxLength(self , arr ):
        # write code here
        #连续，不重复的最长子数组
        #两层循环，一层循环确定最长子数组的头，第二层循环最长子数组的长度。
        maxlen=0
        for i,ea in enumerate(arr):
            for j in range(1,len(arr)-i+1):
                if self.norepeat(arr[i:i+j]):
                    if maxlen<j:
                        maxlen=j
                else:
                    continue
        return maxlen

    def maxLength1(self, arr):
        # write code here
        # 连续，不重复的最长子数组
        # 两层循环，一层循环确定最长子数组的头，第二层循环最长子数组的长度。
        ##连续+不重复，重复不重复由HASH来判断，重复就改变最长子数组的头
        maxlen = 1
        arr_ele = dict()
        start, end = 0, 0
        while end < len(arr):
            if arr[end] in arr_ele.keys():
                # 重复了，改变子数组的头Index,因为不能有重复的，所以新的头index得是重复元素的下一个
                start = max(start, arr_ele[arr[end]] + 1)
            maxlen = max(maxlen, end - start + 1)  # 判断当前的长度
            arr_ele[arr[end]] = end  # 找一个哈希，存的是元素值和对应的下标
            end += 1
        return maxlen
#最长回文子串
def ispalindrome( s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    return s[l + 1:r ]#因为python字符串切片，最后一位取不到r，得取到r-1

def longestPalindrome( s):
    # 从中间向两边扩散找最长回文子串
    ressubstr = ''
    for i in range(len(s)):
        s1 = ispalindrome(s, i, i)
        s2 =ispalindrome(s, i, i + 1)
        ressubstr = s1 if len(ressubstr) <= len(s1) else ressubstr
        ressubstr = s2 if len(ressubstr) <=len(s2) else ressubstr
    return ressubstr
#print(longestPalindrome('cbbd'))
#最长回文子串
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s

        max_len = 1
        begin = 0
        # dp[i][j] 表示 s[i..j] 是否是回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True

        # 递推开始
        # 先枚举子串长度
        for L in range(2, n + 1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            for i in range(n):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = L + i - 1
                # 如果右边界越界，就可以退出当前循环
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]

                # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]
#无重复的三树之和为0，排序+双指针   对于有序的数组，使用双指针
def threeSum( num):
        # write code here
        # dfs
        # 这个只能枚举，暴力循环肯定是3层
    result = []
    track = []
    visited = []

    def backtrack(index):
        if len(track) == 3 and sum(track) == 0:
            temp = sorted(track)
            if temp not in result:
                result.append(temp[:])
            return
        if len(track) > 3:
            return
        track.append(num[index])
        visited.append(index)
        if index + 1 < len(num):
            backtrack(index + 1)
        track.pop()
        visited.pop()

        #             for j in range(len(num)):
        #                 if len(track)>3 or j in visited:
        #                     continue
        #                 track.append(num[j])
        #                 visited.append(j)
        #                 backtrack(j)
        #                 track.pop()
        #                 visited.pop()
    for i in range(len(num)):
        backtrack(i)
    return result
# print(threeSum([-2,0,1,1,2]))
#旋转图像
def rotate( matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    # 把每一列的头元素反转
    axisset = set()
    n, m = len(matrix), len(matrix[0])  # 行和列
    for i in range(m):
        l = 0

        r = n - 1
        while l <= r:
            # temp=matrix[l][i]
            # matrix[l][i]=matrix[r][i]
            # matrix[r][i]=temp
            matrix[l][i], matrix[r][i] = matrix[r][i], matrix[l][i]
            l += 1
            r -= 1
    for x in range(m):
        for y in range(x):  # 上三角
            matrix[x][y], matrix[y][x] = matrix[y][x], matrix[x][y]
    # for x in range(m):
    #     for y in range(n):
    #         if (x,y) not in axisset and (y,x) not in axisset:
    #             matrix[x][y],matrix[y][x]=matrix[y][x],matrix[x][y]
    #             axisset.add((x,y))
    #             axisset.add((y,x))
#最长递增子序列
def lengthOfLIS(nums) :
    # dp
    # dp[i]表示nums[0...i]最长递增子序列长度,#dp定义有问题，递增子序列中的递增不是说的index而是值递增
    # 所以得找小于当前值的子序列+1，dp定义，以nums[i]为结尾的递增子序列长度，然后遍历找出最大
    n = len(nums)
    dp = [1 for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
#最长公共子序列
def longestCommonSubsequence( text1, text2):
    # dp
    # base case
    if len(text1) <= 0 or len(text2) <= 0:
        return 0
    n, m = len(text1), len(text2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]
# a=[3,6,1,5,4,2]
# a.sort()#a.sort()会改变a原本的顺序 ,升序,l.sort()只适用于列表，而非任意可以迭代的对象。
# print(a)
# #可以指定关键字排序
# student=[['tom','A',20],['jack','c',18],['andy','b',11]]
# # student.sort(key=lambda student:student[2])
# print(student)
# #用cmp进行排序
# #sorted()适用于任意可以迭代的对象，可以应用于字符串，元组，列表，字典等可迭代对象
# b='python'
# print(sorted(b))#['h', 'n', 'o', 'p', 't', 'y']
# print(b)#python,不会改变源对象
# e={'1':'a','2':'b','0':'c'}
# print(sorted(e.items()))#[('0', 'c'), ('1', 'a'), ('2', 'b')],返回的是tuple+list
# print(e)#{'2': 'b', '0': 'c', '1': 'a'}
#最长回文子序列

def longestPalindromeSubseq( s):
    n=len(s)
    dp=[[0]*n for _ in range(n)]
    for i in range(n):
        dp[i][i]=1
    for x in range(n-1,-1,-1):
        for y in range(x+1,n,1):
            if s[x]==s[y]:
                dp[x][y]=dp[x+1][y-1]+2
            else:
                dp[x][y]=max(dp[x][y-1],dp[x+1][y])
    return dp[0][n-1]
#单词拆分
def wordBreak( s, wordDict):
    n = len(s)
    dp = [False for _ in range(n + 1)]  # 对于字符串的dp好像都有大于1的情况，主要是这里的base case
    dp[0] = True  # base case空字符一定在worddict中

    for i in range(n):
        for j in range(i + 1, n + 1):  # 判断s[i]到s[j-1]之间的字符串在不在worddict中，找到空格所在处，
            # 注意这里枚举的是字符串，怎么枚举的，是枚举开头位置+长度，
            # 字符串的枚举基本上开头位置和长度，子串也是这样，而不是单个字符
            if dp[i] and (s[i:j] in wordDict):
                dp[j] = True

    return dp[-1]

# def get_lines():
#     l=[]
#     with open('file.txt','rb') as f:
#         #return f.readlines()
#         # for i in f:
#         #     yield i#通过生成器，每次只有一行
#         data=f.readlines(60000)
#     l.append(data)
#     yield l#每次读取60000到内存，再往下读
#
# for e in get_lines():
#     #process(e),处理每一行数据,r如果文件大小为10g，但是内存只有4g的话，就不能用f.readlines一次性读入
#     pass
# from mmap import mmap
# def get_lines(fp):
#     with open(fp,'r+') as f:
#         m=mmap(f.fileno(),0)#创建mmap对象，mmap是在进程的内存中开辟一个空间，映射到某个打开的文件，之后所有发生在这块内存上的变化都会被写入文件。
#         #将文件当做byte array处理，硬盘绕过操作系统内核，直接与mmap内存交互。
#         tmp=0
#         for i,char in enumerate(m):
#             if char==b'\n':
#                 yield m[tmp:i+1].decode()#用array切片，获取当前行换行符之前，因为是byte类型，需要decode
#                 tmp=i+1#tmp表示起始位置，从该位置获取下一个转行符之前的内容。
# for i in get_lines('fp_some_huge_file'):
#     print(i)
import os
def print_directory_contents(spath):
    for s_child in os.listdir(spath):
        s_child_path=os.path.join(spath,s_child)
        if os.path.isdir(s_child_path):
            print_directory_contents(s_child_path)
        else:
            print(s_child_path)
#print_directory_contents('G:/study/CloudMusicSimilarMan-master/')#输入文件夹路径，返回该文件夹包括子文件夹的所有文件路径
#子集
def subsets( nums):
    res = []

    def backtrack(start, track):
        res.append(track[:])  # 注意外层和里层如果同用1个名track，会导致每次变化，外层和内层的track同时影响

        for j in range(start, len(nums)):
            track.append(nums[j])
            backtrack(j + 1, track)
            track.pop()

    backtrack(0, [])  # 这里是搞得index,这样下次选择就是排除当前index的其他选项
    return res  # 互不相同的子集，就是对nums的全排列，回溯也就是dfs
#组合
def combine( n, k):
    if k <= 0 or n <= 0: return []
    result = []

    def backtrack(index, track):
        if len(track) == k:
            # temp=sorted(track)
            # if temp not in result:
            result.append(track[:])
            return

        for i in range(index, n + 1):
            track.append(i)
            backtrack(i + 1, track)  # 真的是，要去重的话一定是backtrack传入index，这样每次回溯排除了已经加入track的元素
            track.pop()

    backtrack(1, [])
    return result
#全排列和组合的区别，从树来看，组合的右边树会越来越少，因为选择时要排除之前的，只能选择列表后面的，用start搞索引，i+1
#全排列只是排除和自己重复的，但是不排除列表前面的，所以用contains
def permute( nums):
    result = []
    n = len(nums)

    def backtrack(track):
        if len(track) == n:
            result.append(track[:])
            return
        for i in range(n):
            if nums[i] in track:  # or len(track)>n:
                continue  # 这个是排除已经选过的数组所以用contain，在选择时要排除当前只要之后的数字，传start
            track.append(nums[i])
            backtrack(track)
            track.pop()

    backtrack([])
    return result
#组合求和2
def combinationSum2(candidates, target):
    # 这个dfs得排除自己,用index
    result = []
    candidates.sort()  # 同一个列表里有重复元素，先排序，去除和前面元素一样的分支，所以得排序，不然就是i-1了

    def backtrack(index, track):
        if sum(track) == target and track not in result:
            result.append(track[:])
            return
        if sum(track) > target:
            return
        for i in range(index, len(candidates)):
            if candidates[i] > target:  # 如果一开始就是10，已经大于target就没有必要回溯下去
                break
            if i > index and candidates[i] == candidates[
                i - 1]:  # 这种是因为可能存在重复的元素，同一层的重复元素计算一次就够了，比如[10,1,2,7,6,1,5]里面，已经计算过10后面的1，就不用计算倒数第二个1，这个时候，i肯定大于index
                continue
            track.append(candidates[i])
            backtrack(i + 1, track)    #i+1是排除当前元素
            track.pop()

    backtrack(0, [])
    return result
#组合求和
def combinationSum(candidates, target):
    result = []
    def backtrack(start, track):
        if sum(track) == target and track not in result:
            # temp=sorted(track)
            # if temp not in result:
            #     result.append(temp[:])
            result.append(track[:])
            return
        if sum(track) > target:
            return
        for i in range(start, len(candidates)):  # 从i开始，可以去重，比如start为1之后，
            # 就不会在计算一遍索引为0的分支，因为索引为的分支已经计算一遍了，如果start为1之后，
            # 索引又从0开始的话，就会重复计算分支，例如【2,3,6,7】2的分支里面会计算2,3,6,7，
            # 那么3的分支卡里面就不要再计算2了。这样就不用对所有track都排序
            # if sum(track)>target:
            #     continue#没有用，前面已经遇到就return了
            track.append(candidates[i])
            backtrack(i, track)  # 这里是排除前面分支在后面的重复计算，也就是3的分支不考虑前面的
            track.pop()
    backtrack(0, [])
    return result
#注意画树的时候，去重
'''
如果是[2,3,6,7]这种，可以重复使用列表的元素，2的分支里面会计算2，3,6,7,这个时候，3的分支也会计算2,3,6,7，
为了去重3的分支里的2，可以在for循环里面加一个start，for i in range(start,n)：dfs(i)可以去重这种情况
如果是[10,1,2,7,6,1,5]这种，列表里的元素只可以计算一次，但是列表里有重复元素，那么除了for循环了排除自身
for i in range(start,size):dfs(i+1)；之后为了去重第二个1 和倒数第二个1，本身进行sort()，在for循环里,
if i>start and nums[i]==num[i-1]:continue给它跳过就可以
'''
# a,b=eval(input())
# print(a,b)
#输入日期，判断这一天是这一年的第几天
# import datetime
# def dayofyear():
#     y=eval(input('year'))
#     m=eval(input('month'))
#     d=eval(input('day'))
#     date1=datetime.date(y,m,d)
#     date2=datetime.date(y,1,1)
#     return (date1-date2).days+1
# dayofyear()
#sorted(d.items(),key=lambda x:x[1])
#字典推导式
# d={k:v for (k,v)in iterable}
# a='astr'
# print(a[::-1])#反转字符串
# l1=['b','c','d','c','a','a']
# l2=sorted(set(l1),key=l1.index)
# print(l2)#
# a='dfsfsdf'
# print(len(a))
from collections import defaultdict

def minWindow(s, t):
    # 先移动right指针，找到覆盖t中所有部分的位置，碰到有效字符要valid++,窗口计数器也要++
    # vaild是窗口中含有有效字符（t）满足条件字符的个数
    # 在valid和len(t)相同时，不断移动left指针，更新结果子串的起始index和长度，碰到有效字符要valid--，窗口计数器也要--
    need,window=dict(),dict()
    for c in t:
        #need[c]+=1
        if c not in need.keys():
            need[c]=1
        else:
            need[c]+=1
    left,right,valid=0,0,0
    start,templength=0,100000
    ns=len(s)
    while right<ns:
        tempc=s[right]
        right+=1
        if tempc in need.keys():
            if tempc not in window.keys():
                window[tempc]=1
            else:
                window[tempc]+=1
            if window[tempc]==need[tempc]:
                valid+=1
        while valid==len(need):
            if right-left<templength:
                start=left
                templength=right-left
            tempd=s[left]
            left+=1
            if tempd in need.keys():
                if window[tempd]==need[tempd]:
                    valid-=1
                window[tempd]-=1
    return ' ' if templength==100000 else s[start:start+templength]
# print(minWindow('ADOBECODEBANC','ABC'))
def checkInclusion( s1, s2) :
    # 因为是排列，所以s1不论咋拍，长度都不变，暂停扩大的条件是right-left>=s2.size
    # 当valid==s2.size就是TRUE
    need, window = dict(), dict()
    for c in s1:
        if c not in need.keys():
            need[c] = 1
        else:
            need[c] += 1
    left, right, valid = 0, 0, 0

    while right < len(s2):
        tempc = s2[right]
        right += 1
        if tempc in need.keys():
            if tempc not in window.keys():
                window[tempc] = 1
            else:
                window[tempc] += 1
            if need[tempc] >= window[tempc]:
                valid += 1
        while right - left >= len(s1):#valid==len(need):窗口停止增长要么长度>=字符串，因为长度一定，要么一定包含字符串valid==len(need)
            if valid == len(need):#这里应该是满足need计数器的情况，而不是s1的情况
                return True
            tempd = s2[left]
            left += 1
            if tempd in need.keys():
                if window[tempd] == need[tempd]:
                    valid -= 1
                window[tempd] -= 1
    return False
# print(checkInclusion("abcdxabcde","abcdeabcdx"))
# print(checkInclusion("ab","eidboaoo"))
#无重复字符的最长子串
def lengthOfLongestSubstring(self, s: str) -> int:
    windows = dict()
    left, right = 0, 0
    res = 0
    while (right < len(s)):
        tempc = s[right]
        right += 1
        if tempc not in windows.keys():
            windows[tempc] = 1
        else:
            windows[tempc] += 1
        while windows[tempc] > 1:  # 有重复元素
            tempd = s[left]
            left += 1
            windows[tempd] -= 1
        res = max(right - left, res)
    return res
#找到字符串中所有字母异位词
def findAnagrams( s, p) :
    need, window = dict(), dict()
    left, right = 0, 0
    res = []
    valid = 0
    for c in p:
        if c not in need.keys():
            need[c] = 1
        else:
            need[c] += 1
    while right < len(s):
        tempc = s[right]
        right += 1
        if tempc in need.keys():
            if tempc not in window.keys():
                window[tempc] = 1
            else:
                window[tempc] += 1
            if window[tempc] == need[tempc]:
                valid += 1
        while right - left >= len(p):  # valid==len(p):因为找到字母异位词，不管怎样，长度必须得大于他才可能，
            if valid == len(need):  # 因为记录的是起始索引，只要valid==need.size，就说明是，因为这里窗口缩小是按照字符串长度，而不是包不包含这个字母。
                res.append(left)
            tempd = s[left]
            left += 1
            if tempd in need.keys():
                if window[tempd] == need[tempd]:
                    valid -= 1
                window[tempd] -= 1

    return res

# def longestSubstring( s, k):
#     # window[c]>=k,记下最长子串长度
#     window = dict()
#     res = 0
#     left, right = 0, 0
#     while right < len(s):
#         tempc = s[right]
#         right += 1
#         if tempc not in window.keys():
#             window[tempc] = 1
#         else:
#             window[tempc] += 1
#         while window[tempc] >= k:  # right-left>=k:#
#             if right - left > res:
#                 res = right - left
#             else:
#
#             tempd = s[left]
#
#             # if window[tempd]>=k:
#             #     res=max(res,right-left)
#             left += 1
#             window[tempd] -= 1
#     return res
# print(longestSubstring('ababbc',2))
#将待排序元素根据gap划分为若干组，组内进行直接插入排序，使序列基本有序，gap每次减少一半，直到gap=1的时候，把整个序列进行插入排序完成。
def shell_sort(nums):
    size=len(nums)
    gap=int(size//2)
    while gap>0:
        for i in range(gap,size):
            j=i
            while j>=gap and nums[j-gap]>nums[j]:
                nums[j-gap],nums[j]=nums[j],nums[j-gap]
                j-=gap
        gap=gap>>1
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
#shell_sort(nums)
#print(nums)
#插入排序
def insertsort(nums):
    for i in range(1,len(nums)):#这里是待排序
        for j in range(i,0,-1):#这里是已排序，得从i开始比较j和j-1,把nms[i]插入到已排序的正确位置。
            if nums[j]<nums[j-1]:
                nums[j],nums[j-1]=nums[j-1],nums[j]
#insertsort(nums)
#快速排序,以pivot为基准，比它大的放后面，比它小的放前面。
def quicksort(nums,start,end):
    if start>=end:
        return
    pivot=nums[start]#通过一趟排序将要排序的数据分割成两部分，其中一部分的所有数据都比另一部分的所有数据小
    l,r=start,end
    while l<r:
        while l<r and nums[r]>=pivot:
            r-=1
        nums[l]=nums[r]

        while l<r and nums[l]<pivot:
            l+=1
        nums[r]=nums[l]
    nums[l]=pivot#通过一趟排序，把pivot找准位置
    quicksort(nums,start,l-1)#再按照这个方法对两部分数据分别进行快速排序，递归进行
    quicksort(nums,l+1,end)
#quicksort(nums,0,len(nums)-1)
#选择排序
def select_sort(nums):

    n=len(nums)
    for i in range(n-1):#在未排序序列中找到最小元素，放在排序序列的起始位置
        minindex=i#每次都是选择最小的，所以如果当前元素是最小的，那么下表不应该有变化。
        for j in range(i+1,n):#再从剩下未排序元素中寻找最小元素的下标，放到已经排序序列的末尾。
            if nums[j]<nums[minindex]:
                minindex=j
        if minindex!=i:
            nums[i],nums[minindex]=nums[minindex],nums[i]


def merge(left,right):
    l,r=0,0
    result=[]
    while l<len(left) and r<len(right):#合并左右两个列表，各自有一个头指针，比较两个列表的头指针，谁小结果集里加入，
        if left[l]<=right[r]:
            result.append(left[l])
            l+=1
        else:
            result.append(right[r])
            r+=1
    result+=left[l:]#针对某一列表长度完了，还有一个列表长度大于他，把这个列表剩余的加到结果中，本身列表两个都是有序的，合并成有序列表
    result+=right[r:]
    return result
def mergesort(nums):
    if len(nums)<=1:
        return
    mid=len(nums)>>2#以中点切开，
    left=mergesort(nums[:mid])#不断对左边切分，直到组的长度为1
    right=mergesort(nums[mid:])#不断对右边划分，直到组的长度为1
    return merge(left,right)#合并左右两个列表

#print(nums)

#至少有k个重复字符的最长子串
# def longestSubstring(s,k):
#     if len(s)<k:
#         return 0
#     for c in set(s):
#         if s.count(c)<k:
#             return max(longestSubstring(t,k) for t in s.split(c))
#     return len(s)

def permuteUnique( nums):
    result, track = [], []
    n = len(nums)
    visited=[]
    def backtrack(tempnum):  #得去除自己，但是得算上之前的,就是每次回溯遍历范围不能变，但是不能算已经有标记的元素，就去除了自己,而且还不是标记元素值，是标记下标
        if len(track) == n and track not in result:
            result.append(track[:])
            return
        if len(track) > n:
            return
        for i in range(len(tempnum)):
            if tempnum[i] in visited:
                continue
            track.append(tempnum[i])
            visited.append(tempnum[i])
            # temp = tempnum[i]
            # tempnum.remove(tempnum[i])
            backtrack(tempnum)
            track.pop()
            visited.pop()
            #tempnum.append(temp)

    backtrack(nums)
    return result
#print(permuteUnique([1,2,3]))
from collections import defaultdict

#340. 至多包含 K 个不同字符的最长子串
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: 'str', k: 'int') -> 'int':
        n = len(s)
        if k == 0 or n == 0:
            return 0

        # sliding window left and right pointers
        left, right = 0, 0
        # hashmap character -> its rightmost position
        # in the sliding window
        hashmap = defaultdict()

        max_len = 1

        while right < n:
            # add new character and move right pointer
            hashmap[s[right]] = right
            right += 1

            # slidewindow contains 3 characters
            if len(hashmap) == k + 1:
                # delete the leftmost character
                del_idx = min(hashmap.values())
                del hashmap[s[del_idx]]
                # move left pointer of the slidewindow
                left = del_idx + 1

            max_len = max(max_len, right - left)

        return max_len
#买卖股票的最佳时机
def maxProfit(self, prices):
    # 当添加一个新值时，和已知的相比，就可以推导出新的最值，如果是减少一个值，就必须遍历才能知道最值情况。
    mincost = prices[0]
    result = 0
    for sell in range(1, len(prices)):
        mincost = min(mincost, prices[sell])
        result = max(result, prices[sell] - mincost)
    return result
#反转链表2
# def reverseBetween526(head,left,right):
#     #先遍历找到pre节点和right节点
#     #在截断反转的链表
#     #反转
#     #拼接
#     def reverse526(head):#这个是反转以head为头结点的链表
#         pre=None
#         cur=head
#         while cur:
#             nxt=cur.next
#             cur.next=pre
#             pre=cur
#             cur=nxt
#
#     dummy=ListNode(-1)#设置一个虚head，方便对头结点和之后的节点统一处理
#     dummy.next=head
#     pre=dummy
#     for _ in range(left-1):#先保存left节点的前一个，为了拼接
#         pre=pre.next
#     rightnode=pre
#     for _ in range(right-left+1):#保存right节点，为了反转，
#         rightnode=rightnode.next
#     leftnode=pre.next#保存前后连接，为了反转和拼接
#     curr=rightnode.next
#     pre.next=None#截断原链表，改成子链表
#     rightnode.next=None#保存子链表
#     reverse526(leftnode)
#     pre.next=rightnode#拼接，因为是原地反转，所以需要接rightnode,leftnode已经变成子链表的尾节点了
#     leftnode.next=curr
#     return dummy.next

#希尔排序
# def shell_sort(nums):
#     #先用gap划分为若干组，组内用直接插入排序，达到基本有序，不断gap减小，直到1
#     gap=len(nums)//2
#     while gap>0:
#         for i in range(gap,len(nums)):
#             j=i
#             while j>=gap and nums[j-gap]>nums[j]:
#                 nums[j],nums[j-gap]=nums[j-gap],nums[j]
#                 j-=gap
#         gap=gap//2
#归并排序,先不断地将组里的元素划分，直到组内元素长度为1，然后两两合并
#两两合并，有两个列表left，right，两个指针分别指向两个列表头部，比较元素大小，放到新的结果集中，长度走完了，直接加到结果中
# def merge(left,right):
#     l,r=0,0
#     result=[]
#     while l<len(left) and r<len(right):#得是and而不是or,得同时满足才能比较二者大小
#         if left[l]<=right[r]:
#             result.append(left[l])
#             l+=1
#         else:
#             result.append(right[r])
#             r+=1
#     result+=left[l:]
#     result+=right[r:]
#     return result
# def mergeSort(nums):
#     if len(nums)<=1:
#         return nums
#     mid=len(nums)//2
#     left=mergeSort(nums[:mid])
#     right=mergeSort(nums[mid:])
#     return merge(left,right)
#快速排序,以pivot为基准，比pivot大的放在high，比pivot小的放在low，不断移动high和low，确保pivot在正确位置，
# 并且pivot左部分数据都小于pivot右部分数据，不断递归左右部分
# def quicksort(nums,start,end):
#     if start>=end:#因为是不断递归，遇到base case 要返回
#         return
#     pivot=nums[start]
#     l,h=start,end
#     while l<h:
#         while l<h and nums[h]>=pivot:
#             h-=1
#         nums[l]=nums[h]
#         while l<h and nums[l]<pivot:
#             l+=1
#         nums[h]=nums[l]
#     nums[l]=pivot
#     quicksort(nums,start,l-1)#要排除pivot，这里又不是数组切片，是包含l-1这个位置的。
#     quicksort(nums,l+1,end)
#插入排序,每次从待排序取出的元素放到有序序列的正确位置
# def insertSort(nums):
#     for i in range(1,len(nums)):
#         for j in range(i,0,-1):
#             if nums[j-1]>nums[j]:
#                 nums[j-1],nums[j]=nums[j],nums[j-1]
#选择排序,每次都从待排序中选择最小的放在已排序的末尾

# def selectSort(nums):
#     for i in range(len(nums)-1):
#         minvalue=i#不要一会存值，一会存下标，minvalue就是存下标的
#         for j in range(i+1,len(nums)):#这里是i+1,所以前面是n-1
#             if nums[j]<nums[minvalue]:
#                 minvalue=j
#         if minvalue!=i:
#             nums[minvalue],nums[i]=nums[i],nums[minvalue]
#     return nums

# def maxProfit(k, prices):
#     i = 0
#     res = 0
#
#     for buy in range(1, len(prices)):
#         if prices[buy] > prices[buy - 1]:
#             res += prices[buy] - prices[buy - 1]
#             i += 1
#             if i>k:
#                 break
#
#     return res
# print(maxProfit(k=2,prices=[3,2,6,5,0,3]))
#最长公共子串
def LCS(str1,str2):
    if len(str1)==0 or len(str2)==0:
        return 0
    maxlen=0
    end=0
    n,m=len(str1),len(str2)
    dp=[[0]*m for i in range(n)]
    for i in range(n):
        if str1[i]==str2[0]:
            dp[i][0]=1
    for j in range(m):
        if str2[j]==str1[0]:
            dp[0][j]=1
    for i in range(1,n):
        for j in range(1,m):
            if str1[i]==str2[j]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=0
            if dp[i][j]>maxlen:
                maxlen=dp[i][j]
                end=i
    return str1[end-maxlen+1:end+1]
#print(LCS("1AB2345CD","12345EF"))
def longestcommonsubsequence(text1,text2):
    if len(text1)==0 or len(text2)==0:
        return None
    n,m=len(text1),len(text2)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            if text1[i-1]==text2[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    return dp[n][m]
# print(longestcommonsubsequence("1AB2345CD","12345EF"))
#删除有序链表中重复出现的元素
# def deleteDuplicates( head):
#     # write code here
#     # 先统计出现的次数，再删除
#     # 这里的升序怎么用
#     dummy = ListNode(-1)
#     dummy.next = head
#     pre = dummy
#     cur = pre.next
#     while cur != None:
#         difnode = cur
#         currepeatnum = 0
#         while difnode != None and difnode.val == cur.val:
#             difnode = difnode.next
#             currepeatnum += 1
#         if currepeatnum > 1:
#             pre.next = difnode
#         else:
#             pre = cur
#         cur = difnode
#     return dummy.next
#合并区间
'''
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
'''
#
# def mergee(intervals):
#     # 这种一个一个添加，只需要和当前比较大小，即可知道
#     intervals.sort(key=lambda x:x[0])
#     result=[intervals[0]]
#     for i in intervals:
#         if i[0]<=result[-1][1]:
#             result[-1][1]=max(result[-1][1],i[1])
#         else:
#             result.append(i)
#     return result


# class Solution {
# public:
#     ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
#         ListNode* head=new ListNode(-1);//存放结果的链表
#         ListNode* h=head;//移动指针
#         int sum=0;//每个位的加和结果
#         bool carry=false;//进位标志
#         while(l1!=NULL||l2!=NULL)
#         {
#             sum=0;
#             if(l1!=NULL)
#             {
#                 sum+=l1->val;
#                 l1=l1->next;
#             }
#             if(l2!=NULL)
#             {
#                 sum+=l2->val;
#                 l2=l2->next;
#             }
#             if(carry)
#                 sum++;
#             h->next=new ListNode(sum%10);
#             h=h->next;
#             carry=sum>=10?true:false;
#         }
#         if(carry)
#         {
#             h->next=new ListNode(1);
#         }
#         return head->next;
#     }
# };
# def twomerge(left,right):
#     l,r=0,0
#     llen,rlen=len(left),len(right)
#     result=[]
#     while l<llen and r<rlen:
#         if left[l]<=right[r]:
#             result.append(left[l])
#             l+=1
#         else:
#             result.append(right[r])
#             r+=1
#     result+=left[l:]
#     result+=right[r:]
#     return result
#
#
#
# def mergessort(nums):
#     if len(nums)==1:
#         return nums
#     low=0
#     high=len(nums)
#     mid=low+(high-low)//2
#     left=mergessort(nums[low:mid])
#     right=mergessort(nums[mid:high])
#     return twomerge(left,right)

# for i in range(1,1000):
#     if '5' not in str(i):
#         print(i)
# import sys
# linee = sys.stdin.readline().strip()
# linee= list(map(int, linee.split()))
# for i,j in enumerate(linee):
#     if j==0:
#         linee.pop(i)
#         linee.append(0)
# print(linee)
#零钱兑换二
'''
组合问题，穷举的时候，要按照一定顺序，2,1和1,2是一种情况，而不是两种
在排列问题中，2,1和1,2是两种情况
'''
# def change(amount,coins):
#     dp=[amount+1 for i in range(amount+1)]#金额为i有dp[i]种组合数
#     dp[0]=1
#     for coin in coins:
#         for i in range(coin,amount+1):#从1到amount，还会出现i-coin<0的情况
#             dp[i]+=dp[i-coin]
#     return dp[amount]
# #零钱兑换
# def change1(amount,coins):
#     dp=[amount+1 for i in range(amount+1)]#金额为i需要最少的硬币个数是dp[i]个
#     dp[0]=0
#     for i in range(1,amount+1):
#         for j in coins:
#             if i-j<0:
#                 continue
#             dp[i]=min(dp[i],dp[i-j]+1)
#     return dp[amount]  if dp[amount]!=amount+1 else -1
#下三角
# for l in range(2,6):
#     for i in range(5-l+1):
#         j=l+i-1
#         print(i,j)
#高楼扔鸡蛋
'''
dp含义，dp[k][m]=n 当前有k鸡蛋，至多扔m次，可以测试最多有n层的楼
我们需要的值是m，但是它在索引位置，所以
'''
# def superEggDrop(k,n):
#     dp=[[0 for i in range(n+1)] for j in range(k+1)]
#     m=0
#     while dp[k][m]<n:
#         m+=1
#         for i in range(1,k+1):#选择扔鸡蛋，只有碎和没碎两种情况
#             dp[k][i]=dp[i][m-1]+dp[i-1][m-1]+1
#     return m
#明确状态，当前有k个鸡蛋，N层楼，求最坏的情况下最小的测试次数m
#选择,在i层楼扔鸡蛋，鸡蛋要么是碎了，要么没碎。碎了，k-1,往低处测，i-1;没碎，k往高处测，n-i

# def superEggDrop(k,n):
#     memo=dict()
#     def dp(k,n):
#         if k==1:
#             return n
#         if n==0:
#             return 0
#         if (k,n) in memo.keys():
#             return memo[(k,n)]
#         res = float('INF')
#         l,h=1,n
#         while l<=h:
#             mid=(h+l)//2
#             broke=dp(k-1,mid-1)
#             unbroke=dp(k,n-mid)
#             if broke>unbroke:
#                 h=mid-1
#                 res=min(res,broke+1)
#             else:
#                 l=mid+1
#                 res=min(res,unbroke+1)
#         memo[(k,n)]=res
#         return res
#     return dp(k,n)
#目标和
# def findTargetSumWays( nums, target):
#     result = []
#     track = []
#     if len(nums)==0:
#         return 0
#     def backtrack(nums, start, track):
#         if start == len(nums):
#             if sum(track) == target:
#                 result.append(1)
#             return
#         if start > len(nums):
#             return
#         for i in [-1, 1]:
#             track.append(i * nums[start])
#             backtrack(nums, start + 1, track)
#             track.pop(-1)
#
#     backtrack(nums, 0, track)
#     return len(result)
# print(findTargetSumWays([1,1,1,1,1],3))


#0-1背包问题
# wt=[2,1,3]
# val=[4,2,3]
# def knapsack(w,n,wt,val):
#     dp=[[0 for i in range(n+1)] for j in range(w+1)]
#     for i in range(1,w+1):
#         for j in range(1,n+1):#因为最后你要搞dp[w][n]，而且wt和val的下标从0开始
#             if i-wt[j-1]<0:#当前背包总容量-第i个物品的容量，就是装i-1个物品的剩余容量
#                 dp[i][j]=dp[i][j-1]
#             else:
#                 dp[i][j]=max(dp[i][j-1],val[j-1]+dp[i-wt[j-1]][j-1])
#     return dp[w][n]
# print(knapsack(4,3,wt,val))
#分割等和子集
'''
将该问题转为背包问题，sum=求和nums,只要存在一种方法可以让前n 
个物品放到容量为sum/2的背包，就说明存在一种方法让数组可以分割成两个sum/2的
子集
'''


# def canPartition(nums):
#     #状态压缩
#     # sum_nums = sum(nums)
#     # if sum_nums % 2 != 0:
#     #     return False
#     # n = len(nums)
#     # sum_num = sum_nums // 2
#     # dp = [False for i in range(sum_num + 1)]
#     # dp[0] = True
#     # for i in range(1, n + 1):
#     #     for j in range(sum_num, 0, -1):#因为每个物品只能用一次，如果是正着写的话，意思就是在计算sum_num时，可以无限用物品次数
#             #原本的希望是只依靠前一个dp状态计算当前，正着循环，会让dp的计算不止依赖当前状态。
#     #
#     #         if j - nums[i - 1] >= 0:
#     #             dp[j] = dp[j - nums[i - 1]] or dp[j]
#     # return dp[sum_num]
#     sum_nums = sum(nums)
#     if sum_nums % 2 != 0:
#         return False
#     n = len(nums)
#     sum_num = sum_nums // 2
#     dp = [[False for i in range(sum_num + 1)] for j in range(n + 1)]
#     # for i in range(sum_nums+1):
#     #     dp[0][i]=False
#     for j in range(n + 1):
#         dp[j][0] = True
#     for i in range(1, n + 1):
#         for j in range(1, sum_num + 1):
#             if j - nums[i - 1] < 0:
#                 dp[i][j] = dp[i - 1][j]  # 这里为啥不直接是False，因为有可能dp[i][j-1]是true,加了第j个数到子集就不行,但是这里是选择不装入
#             else:
#                 dp[i][j] = dp[i - 1][j - nums[i - 1]] or dp[i - 1][j]  # 为啥这里是Or,不是and,因为这里只能有一种选择，要么装入，要么不装入
#     return dp[n][sum_num]


# nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
# print(mergessort(nums))
#shell_sort(nums)
#print(mergeSort(nums))
#quicksort(nums,0,len(nums)-1)
# print(selectSort(nums))
# print(nums)
# print((0.6573059558868408+0.6508563756942749+0.3218013346195221)/3)

# def change( amount, coins):
#     n = len(coins)
#     dp = [0 for i in range(amount + 1)]
#     dp[0] = 1
#     for i in range(1, n + 1):
#         for j in range(amount,0,-1):
#             if j - coins[i - 1] >= 0:
#                 dp[j] = dp[j] + dp[j - coins[i - 1]]
#             print(coins[i-1],j,dp[j])
#     return dp[amount]
# amount=5
# coins=[1,2,5]
# res=change(amount,coins)
'''
目标和转为子集问题--背包问题
'''
def findTargetSumWays(nums, target):
    #状态压缩
    # sum_nums = sum(nums)
    # if sum_nums < target or (target + sum_nums) % 2 != 0:
    #     return 0
    # n = len(nums)
    # sum_nums = (target + sum_nums) // 2  # 类似于分割等和子集
    # # dp=[[0 for j in range(sum_nums+1)]for i in range(n+1)]#nums总前i个物品，总量为j,有dp[i][j]种方式装满
    # dp = [0 for i in range(sum_nums + 1)]
    # # base case
    # # for i in range(n+1):
    # #     dp[i][0]=1
    # dp[0] = 1
    # for i in range(1, n + 1):
    #     for j in range(sum_nums, -1, -1):  # 这里的sum_nums不是单纯的sum求和，而是(target+sum(nums))//2,所以存在0
    #         if j - nums[i - 1] < 0:
    #             dp[j] = dp[j]
    #         else:
    #             dp[j] = dp[j] + dp[j - nums[i - 1]]
    # return dp[sum_nums]
    sum_nums = sum(nums)
    if sum_nums < target or (target + sum_nums) % 2 != 0:
        return 0
    n = len(nums)
    sum_nums = (target + sum_nums) // 2  # 类似于分割等和子集
    dp = [[0 for j in range(sum_nums + 1)] for i in range(n + 1)]  # nums总前i个物品，总量为j,有dp[i][j]种方式装满
    # base case
    for i in range(n + 1):
        dp[i][0] = 1
    for i in range(1, n + 1):
        for j in range(0, sum_nums + 1):##这里的sum_nums不是单纯的sum求和，而是(target+sum(nums))//2,所以存在0
            if j - nums[i - 1] < 0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]]
    return dp[n][sum_nums]
'''
打家劫舍1
'''
def rob(nums):
    # 状态，前i个房屋还有金额变化啊
    # dp[i],前i个房屋的最高金额，dp[-1]
    # 选择就是要么偷还是不偷，dp[i]=dp[i-1]or dp[i-1]+nums[i],这样max选择每次基本都选后者有问题，
    # dp[i+1]  dp[i+2]+nums[i]
    # 状态压缩
    n = len(nums)
    dp_i_1, dp_i_2, dp_i = 0, 0, 0
    for i in range(n - 1, -1, -1):#因为base case 是dp[n]=0,所以遍历得从base case 存在的地方搞起
        dp_i = max(dp_i_1, dp_i_2 + nums[i])
        dp_i_2 = dp_i_1
        dp_i_1 = dp_i
    return dp_i
    #dp table
    # n=len(nums)
    # dp=[0  for i in range(n+2)]
    # for i in range(n-1,-1,-1):
    #     dp[i]=max(dp[i+1],dp[i+2]+nums[i])
    # return dp[0]
    #备忘录
    # memo=dict()
    # def dp(nums,start):
    #     n=len(nums)
    #     if start>=n:
    #         return 0
    #     if start in memo.keys():
    #         return memo[start]
    #     res=max(dp(nums,start+1),dp(nums,start+2)+nums[start])
    #     memo[start]=res
    #     return res
    # return dp(nums,0)
'''
打家劫舍 --环形问题
'''
def robTwo(nums):
    # 环形偷窃3种情况，1、头和尾都不偷 2、头偷尾不偷 3、头不偷尾偷
    # 但是针对最高金额，一定是2、3情况比1大，因为选择余地大
    # 状态就是房屋索引，选择就是枪或者不抢
    n = len(nums)
    if n == 1:
        return nums[0]

    def robrange(nums, start, end):
        dp_i_1, dp_i_2, dp_i = 0, 0, 0
        for i in range(end, start - 1, -1):
            dp_i = max(dp_i_1, dp_i_2 + nums[i])
            dp_i_2 = dp_i_1
            dp_i_1 = dp_i
        return dp_i

    return max(robrange(nums, 0, n - 2), robrange(nums, 1, n - 1))

    # n=len(nums)
    # if n==1:
    #     return nums[0]
    # def dp(nums,start,end):
    #     n=len(nums)
    #     if start>n:
    #         return 0
    #     res=0
    #     for i in range(end,start-1,-1):
    #         res=max(dp(nums,start+1,end),dp(nums,start+2,end)+nums[i])#因为dp解决是在不断缩小问题规模，所以如果是dp(nums,start,end+1)就是在扩大规模。
    #     return res
    # return max(dp(nums,0,n-2),dp(nums,1,n-1))

def robThree(root):
    # 状态就是每个节点，选择还是抢和不抢
    if root == None:
        return 0
    memo = dict()

    def dp(root):#这个应该表示当前节点抢到的最高金额
        if root == None:
            return 0
        if root in memo.keys():
            return memo[root]
        do_i = root.val + (0 if root.left == None else dp(root.left.left)) + (
            0 if root.left == None else dp(root.left.right)) + (
                   0 if root.right == None else dp(root.right.left)) + (
                   0 if root.right == None else dp(root.right.right))  # 抢了当前，只能抢下下家就是dp[i+2]+nums[i]
        no_do_i = dp(root.right) + dp(
            root.left)  # 不抢的话，就和下家的最高金额一样，因为right和left不相连，所以可以抢这一层。就是dp[i+1]，但是因为可以抢这一层，有左右两个
        res = max(do_i, no_do_i)
        memo[root] = res
        return res

    return dp(root)
'''
乘积最大子数组
'''
def maxProduct(nums):
    #求出最大值的情况应该只有3种，要么是i之前都是0，nums[i]最大
    #要么是正*当前值最大，另外是负*当前值，负负得正
    #所以如果是看做动态规划的话，转态就是下标，因为只需要两个状态，所以不用定义一个数组，但是定义的话，应该是，下标为i的数组，最大值是dp[i],对于
    #状态转移，f(i-1)*nums[i],只是还需要分正负，所以有上面3种情况
    n=len(nums)
    if n==0:
        return 0
    if n==1:
        return nums[0]
    max_i,min_i,ans=nums[0],nums[0],nums[0]
    for i in range(1,n):
        t=max_i
        max_i=max(max(min_i*nums[i],nums[i]),max_i*nums[i])
        min_i=min(min(t*nums[i],nums[i]),min_i*nums[i])#因为对于当前每个元素都得计算一个最大值和最小值，由于上面已经修改最大值，所以不能用修改后的最大值。
        ans=max(max_i,ans)
    return ans
    # 起始下标加长度进行穷举
    # 状态，就是上面两个
    # if len(nums)==1:
    #     return nums[0]
    # res=0
    # n=len(nums)
    # def product(nums,i,j):
    #     temp=1
    #     if j>len(nums)-1:
    #         return -1
    #     for a in range(i,j+1):
    #         temp*=nums[a]
    #     return temp

    # res=-1
    # for L in range(1,n+1):
    #     for i in range(n):#j-i+1=L
    #         j=L+i-1
    #         tempproduct=product(nums,i,j)
    #         res=max(tempproduct,res)
    # return res
'''
无重叠区间
'''
def eraseOverlapIntervals(intervals):
    revintervals=sorted(intervals,key=lambda x:x[1])
    res=1#因为把第一个区间作为标准
    end=revintervals[0][1]
    for x in revintervals:
        start=x[0]
        if start>=end:
            res+=1
            end=x[1]
    return len(revintervals)-res#把独立的区间删除，剩下的就是重叠的区间，删掉就是无重叠区间
'''
用最少数量的箭引爆气球
'''
def findMinArrowShots(points):
        #先排序，一个记录弓箭位置，一个记数目
        revpoints=sorted(points,key=lambda x :x[1])
        end=revpoints[0][1]
        res=1
        for x in revpoints:
            start=x[0]
            if start>end:
                res+=1
                end=x[1]
        return res
'''
1288.删除被覆盖区间
注意同一起点，按照降序
区间问题，先排序，后画图看区间的相对位置的可能
'''
def removeCoveredIntervals(intervals):
    sorted_inv=sorted(intervals,key=lambda x:x[0])
    for x in range(1,len(sorted_inv)):
        if sorted_inv[x][0]==sorted_inv[x-1][0] and sorted_inv[x][1]>sorted_inv[x-1][1]:
            sorted_inv[x][1],sorted_inv[x-1][1]=sorted_inv[x-1][1],sorted_inv[x][1]
    #合并区间的开始和结束
    over_num=0
    start,end=sorted_inv[0][0],sorted_inv[0][1]
    for i in range(1,len(sorted_inv)):
        if start<=sorted_inv[i][0] and end>=sorted_inv[i][1]:
            over_num+=1
        elif end>=sorted_inv[i][0] and end<=sorted_inv[i][1]:
            end=sorted_inv[i][1]#等于也能合并
        elif end<sorted_inv[i][0]:
            start=sorted_inv[i][0]
            end=sorted_inv[i][1]
    return len(intervals)-over_num
'''
56.合并区间
'''
def merge( intervals):
    if len(intervals) <= 1:
        return intervals
    sorted_inv = sorted(intervals, key=lambda x: x[0])
    res = list()
    res.append(sorted_inv[0])
    for i in range(1, len(sorted_inv)):
        if res[-1][1] >= sorted_inv[i][0]:
            res[-1][1] = max(sorted_inv[i][1], res[-1][1])
        else:
            res.append(sorted_inv[i])
    return res

    # 也只有三种，覆盖，相交，不相关
    # 维持一个合并区间，覆盖就不变，相交，更新区间左右，不相关也更新区间左右
    # if len(intervals)<=1:
    #     return intervals
    # sorted_inv=sorted(intervals,key=lambda x:x[0])
    # # for x in range(1,len(sorted_inv)):
    # #     if sorted_inv[x][0]==sorted_inv[x-1][0] and sorted_inv[x][1]>sorted_inv[x-1][1]:
    # #         sorted_inv[x][1],sorted_inv[x-1][1]=sorted_inv[x-1][1],sorted_inv[x][1]
    # #不用对end排降序，因为只是合并，短的在上，长的在下的话，这种情况就算在第二种情况里面了相交
    # start,end=sorted_inv[0][0],sorted_inv[0][1]
    # res=[[start,end]]
    # for x in range(1,len(sorted_inv)):
    #     if start<=sorted_inv[x][0] and end>=sorted_inv[x][1]:
    #         pass
    #     elif end>=sorted_inv[x][0] and end<=sorted_inv[x][1]:
    #         end=sorted_inv[x][1]
    #         res[-1][1]=end
    #     elif end<sorted_inv[x][0]:
    #         start=sorted_inv[x][0]
    #         end=sorted_inv[x][1]
    #         res.append([start,end])

    # return res
'''
986.区间列表的交集
已排序，只需要通过画图就可以知道区间的相对位置，和何时增加first和second
'''
def interaIntersection(firstList,secondList):
    f,s=0,0
    res=list()
    while f<len(firstList) and s<len(secondList):
        tf=firstList[f]
        ts=secondList[s]
        if tf[0]<=ts[1] and tf[1]>=ts[0]:
            #这个可以通过逆否来推
            #无关的话是 tf[1]<ts[0] or tf[0]>ts[1],逆否即使上面这个条件
            res.append([max(tf[0],ts[0]),min(tf[1],ts[1])])
        if ts[1]<tf[1]:
            s+=1
        else:
            f+=1
    return res
    # # base case，只要有一方为空，那么ans就是空
    # # 搞两个指针，一个指向first,一个指向second，当二者有交集，second不动，first+1；当二者无交集，second +1,first 不动，再比较，去左边的大，右边的小
    # # 我之前把first和second什么时候前进搞错了
    # # if len(firstList)==0 or len(secondList)==0:
    # #     return []
    # first, second = 0, 0
    # res = list()
    # while first < len(firstList) and second < len(secondList):
    #     tf = firstList[first]  # 相交，相切，无关
    #     ts = secondList[second]
    #     if tf[0] <= ts[1] and tf[1] >= ts[0]:
    #         res.append([max(tf[0], ts[0]), min(ts[1], tf[1])])
    #     if ts[1] < tf[1]:
    #         second += 1
    #     if ts[1] >= tf[1]:
    #         first += 1
    #     #     first+=1
    #     # elif tf[0]>ts[1] or tf[1]< ts[0]:
    #     #     second+=1
    # return res
'''
1024.视频拼接
'''
def videoStitching(clips,time):
    #先排序后画图看区间的相对位置
    #因为所需片段的数目最小的话，一定是每次都选时长最长的那个，
    #长了是肯定能剪的,所以先按起点拍升序，相同起点，排降序，
    sorted_clips=sorted(clips,key=lambda x:x[0])
    for x in range(1,len(sorted_clips)):
        if sorted_clips[x-1][0]==sorted_clips[x][0]:
            if sorted_clips[x-1][1]<sorted_clips[x][1]:
                sorted_clips[x-1][1],sorted_clips[x][1]=sorted_clips[x][1],sorted_clips[x-1][1]
    cur_time,next_time=0,0
    i=0
    n=len(clips)
    res=0
    while i<n and sorted_clips[i][0]<=cur_time:
        while i<n and sorted_clips[i][0]<=cur_time:
            next_time=max(next_time,sorted_clips[i][1])
            i+=1
        cur_time=next_time
        res+=1
        if cur_time>=time:
            return res
    return -1
"""
三数之和
排序+双指针的用法还挺多，到底适合哪类题目
"""
def threeSum724(nums):
    #通过左右指针加上自己来判断
    n=len(nums)
    if n<3:
        return []
    nums.sort()
    res=[]
    for i in range(n):
        if nums[i]>0:
            return res
        if i>0 and nums[i]==nums[i-1]:
            continue
        l=i+1
        r=n-1
        while l<r:
            three_sum=nums[i]+nums[l]+nums[r]
            if three_sum==0:
                res.append([nums[i],nums[l],nums[r]])
                while l<r and nums[l]==nums[l+1]:
                    l=l+1
                while l<r and nums[r]==nums[r-1]:
                    r=r-1
                l=l+1
                r=r-1
            elif three_sum<0:
                l=l+1
            else:
                r=r-1
    return res

def fourSum(nums,target):
    if not nums or len(nums)<4:
        return []
    nums.sort()
    n=len(nums)
    res=[]
    for i in range(n-3):
        if i>0 and nums[i]==nums[i-1]:
            continue
        for b in range(n-2):#b要跟着i改变才行b=i+1
            if b>i+1 and nums[b]==nums[b-1]:
                continue
            c=b+1
            d=n-1
            while c<d:
                four_sum=nums[i]+nums[b]+nums[c]+nums[d]
                if four_sum==target:
                    res.append([nums[i],nums[b],nums[c],nums[d]])
                    while c<d and nums[c]==nums[c+1]:
                        c=c+1
                    while c<d and nums[d]==nums[d-1]:
                        d=d-1
                    c=c+1
                    d=d-1
                elif four_sum<target:
                    c=c+1
                else:
                    d=d-1
    return res
import collections
'''
49.字母异位词分组
给字符串排序得到的hash值是惟一的
给字符串计算频次的hash值也是唯一
其他可能存在哈希冲突
这里的defaultdict默认为空的list
list 不可以直接作为mp的keys,tuple才可以
'''
def groupAnagrams(strs):
    mp=collections.defaultdict(list)
    for s in strs:
        key="".join(sorted(s))
        mp[key].append(s)
    return list(mp.values())
    # mp=collections.defaultdict(list)
    # for s in strs:
    #     counts=[0]*26
    #     for i in s:
    #         counts[ord(i)-ord('a')]+=1
    #     mp[tuple(counts)].append(s)
    # return list(mp.values())
    # # 左右指针，相等将其加到数组，并且复位空值
    # if len(strs) < 1:
    #     return [strs]
    # n = len(strs)
    # result = list()
    #
    # # 通过字母的频次来决定二者是否相等
    # def alpha_count(strsl, strsr):
    #     if len(strsl) != len(strsr):
    #         return False
    #     strl=sorted([i for i in strsl],key=lambda i:i[0])
    #     # for i in range(len(strsl)):
    #     #     if strsl[i] not in strsr:
    #     #         return False
    #     # return True
    #     lcount,rcount=dict(),dict()
    #     for i,j in zip(strsl,strsr):
    #         if i not in lcount.keys():
    #             lcount[i]=1
    #         else:
    #             lcount[i]+=1
    #         if j not in rcount.keys():
    #             rcount[j]=1
    #         else:
    #             rcount[j]+=1
    #     for i in lcount.keys():
    #         if i not in rcount.keys() or lcount[i]!=rcount[i]:
    #             return False
    #     return True
    #
    # for i in range(n):
    #     l = i
    #     r = n - 1
    #     temp = list()
    #     if strs[i] != ' ':
    #         temp.append(strs[i])
    #     else:
    #         continue
    #     while l < r:
    #         if strs[l] == ' ' or strs[r] == ' ':
    #             r-=1
    #             continue
    #         if alpha_count(strs[l], strs[r]):
    #             temp.append(strs[r])
    #             strs[r] = ' '
    #         r -= 1
    #     result.append(temp)
    # return result
# strs =["eat", "tea", "tan", "ate", "nat", "bat"]#["ddddddddddg","dgggggggggg"]# ["eat", "tea", "tan", "ate", "nat", "bat"]
# tres=groupAnagrams(strs)
# print(tres)
# while True:
#     try:
#         a=input().split()
#         b=int(a[0])+int(a[1])
#         print(b)
#     except:
#         break
# n=int(input())
# b=list(input().split())
#
# b.sort()
# print(" ".join(str(i) for i in b))
# print(" ".join(b))
'''
子数组的最大累加和问题
'''
def maxsumofSubarray( arr):
    # write code here
    # dp,状态+选择+状态转移方程
    # 状态，最大累加和，存当前索引存在的最大累加和
    # 选择就是要不要加入当前值
    # max(dp[i-1]+arr[i],dp[i-1])
    if len(arr) == 1:
        return arr[0]
    n = len(arr)
    dp = [0 for i in range(n)]  # 当前数的累加和，而不是最大的累加和
    dp[0] = arr[0]  # 当前树的累加和肯定经历过最后一个数，但是最大的累加和不可能是中间的数组，所以是中间的状态，不是dp[-1]
    res = dp[0]
    for i in range(1, n):  # 必须得连续
        if dp[i - 1] <= 0:  # 如果前i-1个最大累加和为负的，那就舍弃，重新选择i为起点，保证了是子数组
            dp[i] = arr[i]
        else:  # 如果前i-1个为正的话，不管是正还是负都可以尝试
            dp[i] = dp[i - 1] + arr[i]
        res = max(res, dp[i])  # 为啥如果定义的是i处治安的最大累加和，而不是-1
    return res

def generateParenthesis(n):
    # n=2[(()),()()]
    # 状态数是括号对数，选择就是在有效的情况下选择左括号还是右括号
    left,right=0,0
    res=list()
    cur_str=''
    def dfs(n,cur_str,left,right):
        if left==right==n:
            res.append(cur_str)
            return
        if right>left:
            return
        if left<n:
            dfs(n,cur_str+'(',left+1,right)
        if right<left and right<n:
            dfs(n,cur_str+')',left,right+1)
    dfs(n,cur_str,0,0)

    # result = []
    # track = []
    #
    # def youxiao(templist):
    #     z, y = 0, 0
    #     for i in templist:
    #         if i == ')':
    #             y += 1
    #         else:
    #             z += 1
    #         if z - y < 0:
    #             return False
    #     return True if z - y == 0 else False
    #
    #
    # def backtrack(z, y, start,track):
    #     if z > n or y > n or z - y < 0:
    #         return
    #     if len(track) == 2 * n:
    #         if youxiao(track):
    #             strtrack = ''.join(track)
    #             # if strtrack not in result:
    #             result.append(strtrack)
    #         return
    #     for i in range(start,n):
    #         track.append('(')
    #         z += 1
    #         backtrack(z, y, i+1,track)
    #         track.pop(-1)
    #         z -= 1
    #         track.append(')')
    #         y += 1
    #         backtrack(z, y, i+1,track)
    #         track.pop(-1)
    #         y -= 1
    #
    # z, y = 0, 0
    # backtrack(z, y, 0,track)
    # return result
#print(generateParenthesis(3))
"""
nsum
"""
def twoNSum(arr,target):
    #arr.sort()
    left,right=0,len(arr)-1
    res=list()
    while left<right:
        temp_sum=arr[left]+arr[right]
        l=arr[left]
        r=arr[right]
        if temp_sum<target:
            while left<right and arr[left]==l:
                left+=1
        elif temp_sum>target:
            while left<right and arr[right]==r:
                right-=1
        else:#认为出现重复的原因是没有跳过所有重复的元素
            res.append([l,r])
            while left<right and arr[left]==l:
                left+=1
            while left < right and  arr[right] == r:
                right-=1
    return res

#print(nSum([1,3,2,1,3,2,1],4))
#[[1, 3], [1, 3], [2, 2]]
"""
refined 3sum
"""
def threeSumPlus(arr,target):
    arr.sort()
    res=list()
    for i in range(len(arr)):
        tmpres=twoNSum(arr[i+1:],target-arr[i])
        if len(tmpres)!=0:
            #res.append([arr[i],target[:]])
            for ta in tmpres:
                tm=[arr[i]]
                for t in ta:
                    tm.append(t)
                res.append(tm)
        while i<len(arr)-1 and arr[i]==arr[i+1]:
            i+=1
    return res

# print(threeSumPlus([-1,0,1,2,-1,-4],0))
def nSum(arr,start,n,target):
    #先排序，可以将nsum拆成n-1sum的问题
    #但是base是2sum
    sz=len(arr)
    res=list()
    if n<2 or sz<n:#意思就是5sum的话，长度只为2，那肯定不行#n<2 or sz<2:
        return res
    if n==2:
        l,r=start,sz-1#每次都从0开始，就会产生重复，例如3sum，第一个元素选定后，不是再从0-sz里选，而是从选定的那个元素之后开始选
        while l<r:
            left,right=arr[l],arr[r]
            tmpsum=left+right
            if tmpsum<target:
                while l<r and arr[l]==left:
                    l+=1
            elif tmpsum>target:
                while l<r and arr[r]==right:
                    r-=1
            else:
                res.append([left,right])
                while l < r and arr[l] == left:
                    l+=1
                while l < r and arr[r] == right:
                    r-=1

    else:
        for i in range(start,sz):
            tmpnsum=nSum(arr,i+1,n-1,target-arr[i])
            if len(tmpnsum)!=0:
                for tp in tmpnsum:
                    tm=[arr[i]]
                    for t in tp:
                        tm.append(t)
                    res.append(tm)
            while i<sz-1 and arr[i]==arr[i+1]:
                i+=1
    return res
tmparr=[-1,0,1,2,-1,-4]
tmparr.sort()
#print(nSum(tmparr,0,3,0))
# maxres=0
# def isbst(node,minval,maxval):
#     if not node:
#         return True
#     return minval<node.val<maxval and isbst(node.left,minval,node.val) and isbst(node.right,node.val, maxval)
# def sumbst(node):
#     global maxres
#     if not node:
#         return 0
#     cur_sum=node.val+sumbst(node.left)+sumbst(node.right)
#     maxres=max(cur_sum,maxres)
#     return cur_sum
# def findmaxsumbst(node):
#     if isbst(node,-0x3f3f3f3f, 0x3f3f3f3f):
#         sumbst(node)
#         return
#     findmaxsumbst(node.left)
#     findmaxsumbst(node.right)
#
# def maxSumBST(root):
#     findmaxsumbst(root)
#     return maxres
def maxsumbst(root):
    maxres=0
    def traverse(node):
        if not node:
            return [1,0x3f3f3f3f,-0x3f3f3f3f,0]
        left=traverse(node.left)
        right=traverse(node.right)
        res = [0, 0, 0, 0]
        if left[0]  and right[0] and left[2]<node.val<right[1]:
            res[0]=1
            res[1]=min(left[1],node.val)
            res[2]=max(right[2],node.val)
            res[3]=node.val+left[3]+right[3]
            maxres=max(maxres,res[3])
        else:
            res[0]=0
        return res

    traverse(root)
    return maxres
















