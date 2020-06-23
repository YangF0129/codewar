def getCount(inputStr):
    return sum(1 for let in inputStr if let in "aeiouAEIOU")
#######################################################

def binary_array_to_number(arr):
  return int("".join(map(str, arr)), 2)   

########################################################
# test 正确 submit 通过不了
def pig_it(test):
    output_list=[]
    for text in test.split(' '):
        if (text).isalpha():
            text_1=text[:1]
            text=text.replace(text[:1], '')
            output=list(text)+list(text_1)+list('ay')
            output=''.join(output)
            output_list.append(output)
        else:
            output_list.append(text)
    
    return (" ".join(output_list))


#################################################################

def find_outlier(int):
    odds = [x for x in int if x%2!=0]
    evens= [x for x in int if x%2==0]
    return odds[0] if len(odds)<len(evens) else evens[0]

##################################################################
#task playing with digits enumerate（str) return (index, str)

def dig_pow(n, p):
  s = 0
  for i,c in enumerate(str(n)):
     s += pow(int(c),p+i)
  return s/n if s%n==0 else -1

import math
def dig_pow(n, p):
    y=0
    for i in range(len(list(str(n)))):
        y +=math.pow(int(list(str(n))[i]),int(p+i))
    return (int(y/n) if y%n==0 else -1  )
print(dig_pow(89, 1))


#########################################################
#sort the odd pop() 一个一个弹出来啥哈哈哈
def sort_array(arr):
  odds = sorted((x for x in arr if x%2 != 0), reverse=True)
  return [x if x%2==0 else odds.pop() for x in arr]

import numpy as np
def sort_array(source_array):
    odd=[x for x in source_array if x%2!=0]
    odd=np.sort(odd)
    j=0
    for i in range(len(source_array)):
        if source_array[i]%2!=0:
            source_array[i]=odd[j]
            j+=1
    return source_array
    # Return a sorted array. 
print(sort_array([5, 3, 2, 8, 1, 4])) 

##############################################
#moving zeros to end !!! remove 会把False  视为 0
def move_zeros(arr):
    l = [i for i in arr if isinstance(i, bool) or i!=0]
    return l+[0]*(len(arr)-len(l))

############################################
#Duplicate Encoder  使用 for +{}.feq(i,0)+1 对string 或者list里面元素进行计数
###### string list +>count()/////{}.get(c,0)+1////enumerate()
###### string replace 
###### list+list
def duplicate_encode(word):
    freq = {}
    char_list = list(word.lower())
    for c in word.lower():
        freq[c] = freq.get(c,0)+1
    for i,c in enumerate(char_list):
        if(freq[c]==1):
            char_list[i]='('
        else:
            char_list[i]=')'
    return ''.join(char_list)


def duplicate_encode(word):
    return "".join(["(" if word.lower().count(c) == 1 else ")" for c in word.lower()])
#####################################################################
#Dubstep  1，如何将多个连续空格转成一个 2.string.replace(a,b,n) 将前n 个a 转成b

def song_decoder(song):    
    song=song.replace("WUB"," ")
    song=" ".join(song.split())
    return song
##############################################
##类fibonaci
def tribonacci(signature, n):
  res = signature[:n]    ###如果n<3 ,返回前n-1个元素形成的list ！！！
  for i in range(n - 3): res.append(sum(res[-3:]))
  return res

###############################################
##filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
def order(sentence):
    sentence=sentence.split()
    sentence_1=[]
    for i in range(len(sentence)):
       for x in sentence:
           if str(i+1) in x:
               sentence_1.append(x)
    sentence_1=" ".join(sentence_1)
               
  
    return sentence_1

def order(sentence):
    return " ".join(sorted(sentence.split(), key=lambda x: int(filter(str.isdigit, x))))


###############
#Regex Password Validation
from re import compile,VERBOSE
regex = compile("""
^              # begin word
(?=.*?[a-z])   # at least one lowercase letter
(?=.*?[A-Z])   # at least one uppercase letter
(?=.*?[0-9])   # at least one number
[A-Za-z\d]     # only alphanumeric
{6,}           # at least 6 characters long
$              # end word
""", VERBOSE)

###############################
#encode Rot13
def rot13(message):
    t = ""
    for c in message:
        if c.islower(): #str是可以直接比较的
            t += chr( ord('a') + ((ord(c)-ord('a')) + 13 )%26 )
        elif 'A'<=c<='Z':
            t += chr( ord('A') + ((ord(c)-ord('A')) + 13 )%26 )
        else:
            t += c
    return t


    
####################################
#Help the bookseller !

def stock_list(listOfArt, listOfCat):
    if (len(listOfArt) == 0) or (len(listOfCat) == 0):
        return ""
    result = ""
    for cat in listOfCat:
        print(cat)
        total = 0
        for book in listOfArt:
            if (book[0] == cat):
                total += int(book.split(" ")[1])
        if (len(result) != 0):
            result += " - "
        result += "(" + cat + " : " + str(total) + ")"
    return result


def stock_list(stocklist, categories):
    if not stocklist or not categories:
        return ""
    return " - ".join(
        "({} : {})".format(
            category,
            sum(int(item.split()[1]) for item in stocklist if item[0] == category))
        for category in categories)
#我的代码，如何结局dictionary 的key 带引号的问题
def stock_list(b,c):
    for cat in c:
        print(cat)

    dict = {cat: 0 for cat in c}
    for cat in c:
        for x in b:
            if x[0] ==cat:
                dict[cat]+= int(x.split(' ')[1])
    return (str('-'.join(str(d) for d in dict.items())))
b = ["ABAR 200", "CDXE 500", "BKWR 250", "BTSQ 890", "DRTY 600"]
c = ["A", "B"]
print(stock_list(b,c))


##############################################
def iq_test(numbers):
    e = [int(i) % 2 == 0 for i in numbers.split()]

    return e.index(True) + 1 if e.count(True) == 1 else e.index(False) + 1
#############################################
def solution(s):
    return "".join(" "+c if c.upper() else c for c in s)
######################################
#rainfall
data = """Rome:Jan 81.2,Feb 63.2,Mar 70.3,Apr 55.7,May 53.0,Jun 36.4,Jul 17.5,Aug 27.5,Sep 60.9,Oct 117.7,Nov 111.0,Dec 97.9
London:Jan 48.0,Feb 38.9,Mar 39.9,Apr 42.2,May 47.3,Jun 52.1,Jul 59.5,Aug 57.2,Sep 55.4,Oct 62.0,Nov 59.0,Dec 52.9
Paris:Jan 182.3,Feb 120.6,Mar 158.1,Apr 204.9,May 323.1,Jun 300.5,Jul 236.8,Aug 192.9,Sep 66.3,Oct 63.3,Nov 83.2,Dec 154.7
NY:Jan 108.7,Feb 101.8,Mar 131.9,Apr 93.5,May 98.8,Jun 93.6,Jul 102.2,Aug 131.8,Sep 92.0,Oct 82.3,Nov 107.8,Dec 94.2
Vancouver:Jan 145.7,Feb 121.4,Mar 102.3,Apr 69.2,May 55.8,Jun 47.1,Jul 31.3,Aug 37.0,Sep 59.6,Oct 116.3,Nov 154.6,Dec 171.5
Sydney:Jan 103.4,Feb 111.0,Mar 131.3,Apr 129.7,May 123.0,Jun 129.2,Jul 102.8,Aug 80.3,Sep 69.3,Oct 82.6,Nov 81.4,Dec 78.2
Bangkok:Jan 10.6,Feb 28.2,Mar 30.7,Apr 71.8,May 189.4,Jun 151.7,Jul 158.2,Aug 187.0,Sep 319.9,Oct 230.8,Nov 57.3,Dec 9.4
Tokyo:Jan 49.9,Feb 71.5,Mar 106.4,Apr 129.2,May 144.0,Jun 176.0,Jul 135.6,Aug 148.5,Sep 216.4,Oct 194.1,Nov 95.6,Dec 54.4
Beijing:Jan 3.9,Feb 4.7,Mar 8.2,Apr 18.4,May 33.0,Jun 78.1,Jul 224.3,Aug 170.0,Sep 58.4,Oct 18.0,Nov 9.3,Dec 2.7
Lima:Jan 1.2,Feb 0.9,Mar 0.7,Apr 0.4,May 0.6,Jun 1.8,Jul 4.4,Aug 3.1,Sep 3.3,Oct 1.7,Nov 0.5,Dec 0.7"""

towns = ["Rome", "London", "Paris", "NY", "Vancouver", "Sydney", "Bangkok", "Tokyo",
         "Beijing", "Lima", "Montevideo", "Caracas", "Madrid", "Berlin"]
def get_towndata(town, strng):
    for x in strng.split("\n"):# 把data进行逐行分割
        town_1, date= x.split(":")# ！！！将每一行用“：”进行分割，分割后分别传入两个数据
        if town_1==town:
            return [i.split(" ") for i in date.split(",")] 
    return None
##return的新式是[[月份1，降雨量1]，[月份2，降雨量2]，。。。。]

def mean(town,strng):
    data=get_towndata(town,strng)
    if data!=None:
        return sum(float(x) for y,x in data)/len(data)
    else:
        return -1.0

def variance(town,strng):
    data=get_towndata(town,strng)
    mean_1=mean(town,strng)
    if data is not None:
        return sum([(float(x)-mean_1)**2 for m,x in data]) / len(data)
    else:
        return -1.0

print(mean("London", data))
print(variance("London",data))
##################用dictionary 进行分割  ！！ strng.splitlines()等价于strng.split("\n")
import numpy as np

def split_data(town,strng):
    city_data = dict(city.split(":") for city in strng.splitlines())
    return dict(v.split() for v in [m for m in city_data[town].split(',')]) if town in city_data else 0
#return形式
#{'Jan': '48.0', 'Feb': '38.9', 'Mar': '39.9', 'Apr': '42.2', 。。}
def mean(town, strng):
    d = split_data(town, strng)
    return np.average(np.array([float(x) for x in d.values()])) if d else -1
    
def variance(town, strng):
    d = split_data(town, strng)
    return np.var(np.array([float(x) for x in d.values()])) if d else -1

####################用正则表达式
import re
import statistics

def parse(town, strn):
    line = next((line for line in strn.splitlines() if line.startswith(town + ':')), '')
    return list(map(float, re.findall(r'\d+(?:\.\d+)?', line)))
#rain = lambda town, strng: map(float, re.findall("\d+(?:\.\d+)?", "".join(re.findall(town+":(.+)\n", strng))))
# data_1=re.findall(town+":(.+)\n", strng)[0] 
#   rain= [float(x) for x in re.findall("\d+(?:\.\d+)?",data_1)]
#    
def mean(town, strng):
    xs = parse(town, strng)
    return statistics.mean(xs) if xs else -1

def variance(town, strng):
    xs = parse(town, strng)
    return statistics.pvariance(xs) if xs else -1

#########################################################
##more about正则表达式

import re
 
str = 'aabbabaabbaa'
#一个"."就是匹配除 \n (换行符)以外的任意一个字符
print(re.findall(r'a.b',str))#['aab', 'aab']
#*前面的字符出现0次或以上
print(re.findall(r'a*b',str))#['aab', 'b', 'ab', 'aab', 'b']
#贪婪，匹配从.*前面为开始到后面为结束的所有内容
print(re.findall(r'a.*b',str))#['aabbabaabb']
#非贪婪，遇到开始和结束就进行截取，因此截取多次符合的结果，中间没有字符也会被截取
print(re.findall(r'a.*?b',str))#['aab', 'ab', 'aab']
#非贪婪，与上面一样，只是与上面的相比多了一个括号，只保留括号的内容
print(re.findall(r'a(.*?)b',str))#['a', '', 'a']
str = '''aabbab
         aabbaa
         bb'''#后面多加了2个b
#没有把最后一个换行的aab算进来
print(re.findall(r'a.*?b',str))#['aab', 'ab', 'aab']
#re.S不会对\n进行中断
print(re.findall(r'a.*?b',str,re.S))#['aab', 'ab', 'aab', 'aa\n         b']

############################################
#############################################
#buid tower
# [
#   '     *     ', 
#   '    ***    ', 
#   '   *****   ', 
#   '  *******  ', 
#   ' ********* ', 
#   '***********'
# ]
def tower_builder(n):
    return [("*" * (i*2-1)).center(n*2-1) for i in range(1, n+1)]
####  re
def tower_builder(n):
    length = n * 2 - 1
    return ['{:^{}}'.format('*' * a, length) for a in xrange(1, length + 1, 2)]
############################################
#################################################
#String incrementer
def increment_string(strng):
    if strng=="":
        strng="1"
    elif strng[-1].islower() or strng.isupper():
       strng=strng+"1"
    else:
        i=-1
        number=""
        while abs(i)<len(strng) and strng[i].isdigit() :
            number=strng[i]+number
            i=i-1
        n=len(number)
        number=str(int(number)+1)
        if len(number)<n:
            number="0"*(n-len(number))+number
        strng=strng[0:-n]+number   
    return strng
#standard solution
def increment_string(strng):
    head = strng.rstrip('0123456789')# rstirp 取右侧的数字
    tail = strng[len(head):]
    if tail == "": return strng+"1"  #if return 一行
    return head + str(int(tail) + 1).zfill(len(tail)) ##str.zfill(length)  str扩成length 不足用0补


############################################################
###########################################################
#Unary function chainer
#chained([a,b,c,d])(input)=> d(c(b(a(input))))   abcd为function
def chained(functions):
    def f(x):
        for function in functions:
            x = function(x)
        return x
    return f

 ##########################
#  Sum of Pairs 超时警告 使用set() 集合（set）是一个无序的不重复元素序列
def sum_pairs(lst, s):
    cache = set()
    for i in lst:
        if s - i in cache:
            return [s - i, i]
        cache.add(i)

def sum_pairs(ints, s):
    seen=[]
    for item in ints:
        if s-item in seen: return [s-item, item]
        if item not in seen: seen+=[item]
    return None
 
def sum_pairs(ints, s):
    result_index=len(ints)
    for i,x in enumerate(ints):
        if ints[i+1:].count(s-x)>0 and i!=len(ints) and ints[i+1:].index(s-x)+i +1<result_index:
           result_index=ints[i+1:].index(s-x)+i +1
    print(result_index)
    if result_index==len(ints):
        return None
    else:
        return [s-ints[result_index],ints[result_index]]

l1= [1, 4, 8, 7, 3, 15]
print(sum_pairs(l1, 10))


#################################
##########################
#A Rule of Divisibility by 13
array = [1, 10, 9, 12, 3, 4]
def thirt(n):
    total = sum([int(c) * array[i % 6] for i, c in enumerate(reversed(str(n)))])
    if n == total:
        return total
    return thirt(total)

def thirt(n):
    list_1=[1, 10, 9, 12, 3, 4]
    while n>99:
        list_n=[int(x) for x in str(n)]
        n=0  #!!!!想到return def()
        n=sum([int(c) * array[i % 6] for i, c in enumerate(reversed(str(n)))])
        # for i,x in enumerate(reversed(list_n)):
        #     n+=x*list_1[(i%6)]
        print(n)
    return n
##########################################
#Multi-tap Keypad Text Entry on an Old Mobile Phone
#find() 方法检测字符串中是否包含子字符串 str ，如果指定 beg（开始） 和 end（结束） 范围\\，
# 则检查是否包含在指定范围内，如果包含子字符串返回开始的索引值，否则返回-1

BUTTONS = [ '1',   'abc2',  'def3',
          'ghi4',  'jkl5',  'mno6',
          'pqrs7', 'tuv8', 'wxyz9',
            '*',   ' 0',    '#'   ]
def presses(phrase):
    return sum(1 + button.find(c) for c in phrase.lower() for button in BUTTONS if c in button)

def presses(phrase):
    x = 0
    for letter in phrase:
        if letter.lower() in list('1*#adgjmptw '): x+= 1
        elif letter.lower() in list('0behknqux'): x+= 2
        elif letter.lower() in list('cfilorvy'): x+= 3
        elif letter.lower() in list('234568sz'): x+= 4
        elif letter.lower() in list('79'): x+= 5
    return x
#################
#Directions Reduction

def dirReduc(arr):
    dir = " ".join(arr)
    dir2 = dir.replace("NORTH SOUTH",'').replace("SOUTH NORTH",'').replace("EAST WEST",'').replace("WEST EAST",'')
    dir3 = dir2.split()
    return dirReduc(dir3) if len(dir3) < len(arr) else dir3


opposite = {'NORTH': 'SOUTH', 'EAST': 'WEST', 'SOUTH': 'NORTH', 'WEST': 'EAST'}
def dirReduc(plan):
    new_plan = []
    for d in plan:
        if new_plan and new_plan[-1] == opposite[d]:
            new_plan.pop()
        else:
            new_plan.append(d)
    return new_plan


def dirReduc(arr):
    opposites = [{'NORTH', 'SOUTH'}, {'EAST', 'WEST'}]
    
    for i in range(len(arr)-1):
        print(i)
        if set(arr[i:i+2]) in opposites:
            print(arr)
            del arr[i:i+2]
            print(arr)
            return dirReduc(arr)
    return arr   
print(dirReduc(["NORTH","NORTH","SOUTH","SOUTH"]))
###################################
#如何使用try except .raiseTypeError,
# 如何处理可能有可能没有的parameter ,return undeifined when there is None 
def prefill(n=0,v=None):
    try:
        return [v] * int(n)
    except:
        raise TypeError(str(n) + ' is invalid')


################################
#Vasya - Clerk
def tickets(people):
    moneyonhand=[]
    for i,x in enumerate(people):
        if x==25:
            moneyonhand.append(25)
        elif x==50 :
            if moneyonhand.count(25)>=1:
                moneyonhand.remove(25)
                moneyonhand.append(50)
            else:
                return "NO"        
        elif x==100 :
            if moneyonhand.count(50)>=1 and moneyonhand.count(25)>=1:
                moneyonhand.remove(50)
                moneyonhand.remove(25)
                
            elif moneyonhand.count(25)>=3:
                moneyonhand.remove(25)
                moneyonhand.remove(25)
                moneyonhand.remove(25)
            else:
                return "NO"             
    return "YES"           
print(tickets([25, 25, 50]))


def tickets(a):
    n25 = n50 = n100 = 0
    for e in a:
        if   e==25            : n25+=1
        elif e==50            : n25-=1; n50+=1
        elif e==100 and n50>0 : n25-=1; n50-=1
        elif e==100 and n50==0: n25-=3
        if n25<0 or n50<0:
            return 'NO'
    return 'YES'
####################
def alternate_sort(l):
    a = sorted([i for i in l if i >= 0])[::-1]
    print(a)
    b = sorted([i for i in l if i not in a])
    print(b)
    res = []
    while len(a) or len(b):
        if len(b):
            res.append(b.pop())
        if len(a):
            res.append(a.pop())
    return res
print(alternate_sort([5, 2, -3, -9, -4, 8,-42]))
#sorting the elements ascending by their absolute value
import heapq
def alternate_sort(l):
    q=[]
    for i,x in enumerate(l):
        heapq.heappush(q,(abs(x),x))
    result= [heapq.heappop(q)[1] for i in range(len(q))]
print(alternate_sort([5, 2, -3, -9, -4, 8,-42]))
