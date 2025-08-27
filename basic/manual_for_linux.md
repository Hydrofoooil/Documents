## linux文件系统

### 目录结构

linux不像windows那样分盘，而是将整个磁盘按树状结构管理。

> `根目录(/)`
>
> - `bin`--是binary的缩写，用于存放常用命令，如`ls`,`cp`, `mv`等，每个命令对应一个程序
> - `boot`--用于存放linux启动时需要的核心文件
> - `dev`--是dvice的缩写，主要存放一些linux的设备文件，比如外设端口
> - `etc`--主要存放系统用户的配置文件和子目录
> - `home`--主要用来存放用户目录
> - `lib`--是library的缩写，主要用于存放一些动态库，供应用程序调用
> - `lost+found`--一般是空的，当系统异常关机后，相关文件会存放在此目录里
> - `media`--自动挂载一些linux系统自动识别的设备，比如U盘、光驱、windows里的系统盘等
> - `mnt`--提供给用户的用于挂载的临时别的文件系统（手动挂载），比如另外的硬盘
> - `opt`--提供给主机额外安装软件所需要的目录
> - `proc`--是虚拟的目录，是系统内存的映射，可以通过直接访问这个目录来获取系统信息
> - `root`--超级用户的主目录
> - `sbin`--s是super user的简称，此目录主要存放一些系统管理员所用到的系统命令
> - `srv`--主要用来存放一些系统服务启动之后用到的程序
> - `run`--主要用来存放一些系统运行时用到的程序
> - `usr`--主要用于一些用户的应用程序和文件，类似于`opt`和windows下的`program files`
>   - `bin`--存放系统用户所使用的应用程序
>   - `sbin`--存放超级用户使用的高级程序和系统守护程序
>   - `src`--内核源代码默认的存放目录
> - `tmp`--存放一些临时文件
> - `var`主要存放一些经常被修改的文件，如电子日志、邮件等
> - `...`

### 终端文字颜色的含义

<img src="/home/maoting/snap/typora/96/.config/Typora/typora-user-images/image-20250717010111971.png" alt="image-20250717010111971" style="zoom: 33%;" />

另外：红字灰底的文字代表已经失效的链接文件

## 常用终端快捷键

`ctrl`+`a`光标移到行首

`ctrl`+`e`光标移到行末

`ctrl`+`p`翻到上一页

`ctrl`+`n`翻到下一页

`ctrl`+`u`删除光标至开始的全部内容

`ctrl`+`k`删除光标至末尾的全部内容

`ctrl`+`w`删除光标前单词

## 文件权限

```bash
drwxr--rx-
```

第一个字母d后面有3组3个字符，3组从前往后分别所有者、用户组和其他人对于该文件的权限。

每组每组可能出现的字符为：`r` （可读）`x`（可执行） `w`（可写） `-`（无权限）。

- 所有者：即文件的创建者
- 用户组：根据一定规则分组
- 其他人：除了所有者和用户组之外的人

## 通配符与目录符

通配符可以用在任何表示文件名的位置

`*` ：匹配0个或多个字符串，**在文件操作中也代表目录下的所有文件**

`?`：匹配1个字符

`[abcd]`：匹配abcd中的任意一个字符

`[a-z]`：匹配a-z范围里的任意一个字符

`[!abc]`：不匹配abc中的任意一个字符

`/`：根目录

`~`：当前用户的家目录

`.`：当前目录

`..`：上级目录，`../..`即为上两级目录，以此类推

`-`：上一次所在的目录

## 帮助命令

### `man`

`man`是系统自带的参考手册（manual），常用指令如下：

```bash
man man #在终端调出参考手册
```

```bash
man <command> #查看该命令的所有帮助信息(若重名则返回第一个)
```

```bash
man -f <command> #查看该命令的简短描述信息
```

```bash
man -k <keyword> #根据关键词模糊搜索整个手册
```

```bash
man man
/<keyword> #逐个查找关键字（高亮显示）
           #按N查找下一个，按shift+N查找上一个
```

```bash
man <chapter> <command> #在指定章节搜索命令的帮助信息(不同章节有重名的命令)
```

```bash
man -w <command> #搜索该命令对应的手册的存储位置
```

### `info`

`info`命令可以用来阅读`info`格式的文档来查看帮助信息。info文件是由各个节点构成的树状结构，可以像浏览网页般跳转

常用操作：

`q`：退出

`return`：光标移到节点的链接后按回车(return)，跳转到对应链接

`l`：(last)跳转上一个访问的节点，类似浏览器中的“后退”

`n`：(next)跳转同一层的后一节点

`p`：(previous)跳转同一层的前一节点

`u`：(up)跳转父节点

`d`：跳转根节点(top)

`b`/`e`：来到节点的开始/结束

### `whatis`

用来查询某个命令的作用，等同于`man -f`

```bash
whatis <command>
```

## 目录操作

### `cd`

结合目录符，进入目录

### `pwd`

是print working directory的缩写，打印当前所在目录的绝对路径

### `ls`

显示目录信息

```bash
ls <directory> #显示对应目录内的文件
```

```bash
ls -l #逐行列出每个文件的详细信息
ls -R #递归列出所有子目录
ls f* #列出f开头的所有文件或目录
ls -h #自动使用便于阅读的单位来显示数字大小

ls -a #显示所有文件及目录，包含'.'开头的隐藏文件、当前目录（'.'）和父目录（'..'）
ls -A #同上，但不列出当前目录和父目录
ls -r #反序显示文件（默认为英文字典序）
ls -t #按最后修改的时间排序
ls -S #按文件大小排序
ls -F #在列出的file后加同一的符号，例如可执行文档加'*'，目录加'/'
```

### `tree`

```bash
tree -a #显示所有文件包括隐藏文件
tree -L <layers> #显示指定层数
tree -t #按最后修改的时间排序
tree -r #按最后修改的时间的倒序
tree -d #只显示directory称
tree -f #显示时带上完整的相对路径名称
tree -p #显示权限信息
```

## 文件增删

### `touch`

改变已有文件的时间戳属性，或者创建新的文件

```bash
touch <file1> <file2> ...... <filen> #将已有文件的时间戳改为当前时间（前提是拥有文件的写权限），或创建新文件
```

自定义时间戳

```bash
touch -d "yesterday" <file> #将文件的时间戳全部改为昨天同一时间，"yesterday"也可改为"tommorow" "next month"等等
touch -t 2201011030.55 <file> #将文件时间戳改为2022年1月1日10:30分55秒
```

文件有`Access` `Modify` `Change`三个时间戳，若不加参数则默然三个一起修改，否则需要加上对应参数（仅对于`Access`和`Modify`而言）

```bash
touch -a <file> #将Modify时间戳改为当前时间
touch -m <file> #将Modify时间戳改为当前时间
```

若需要在改变时间戳时避开创建新文件

```bash
touch -c <file> #若文件存在，改变时间戳；若文件不存在，不创建新文件且不报错
touch --no-create <file> #同上
```

将其他文件的时间属性复制给到当前文件

``` bash
touch <file1> -r <file2> #将文件2的时间戳复制给文件1
```

### `mkdir`

当已存在同名文件或目录时，会创建目录失败并报错

```bash
mkdir <directory1> <director2> ... <directoryn> #创建多个目录
mkdir dir{5..7} #创建dir5到dir7
```

递归创建目录

```bash
mkdir -p dir1/dir2/dir3 #三个目录为嵌套关系
```

`mkdir`默认创建的目录权限为775（`rwxrwxr-x`），但也可以在建立目录的同时设置目录的权限

```bash
mkdir -m 700 dir1/dir10 #创建dir10，并将其权限设为700（rwx------)
```

在创建时也可逐行输出创建的过程，便于检查

```bash
mkdir -v dir{11..13} 
```

### `rm`

直接删除目录或文件，不放回回收站

```bash
rm <file> #只能删除文件，对于目录会报错
rm -r <directory> #递归删除目录及包含的所有内容，对于只读文件会提示是否确认删除
rm -f <file> #强制删除，忽略不存在的文件或目录，忽略是否只读，不报错
rm -i <file> #删除前会逐个文件询问用户是否执行操作
```

**不同的参数也可以组合使用，下同**

```bash
rm -rf * #强制删除当前目录下的所有内容
rm -ri <目录>
```

### `rmdir`

只删除空目录，一般用于节省磁盘空间

```bash
rmdir <directory>
rmdir -p dir1/dir2/dir3/dir4 #递归删除dir1到dir4（四者为嵌套关系）
rmdir -v <directory> #在创建时逐行输出创建的过程，便于检查
```

### `mv`

用于移动文件或重命名文件

```bash
mv <file/directory> <directory> #将文件/目录移至对应目录下
mv <file1/directory1> <file2/directory2> #将文件/目录重命名
```

**注意：**系统并不知道输入的名称是directory还是file。对于输入的第二个名称，系统会优先匹配是否有该名称的目录，若有则将第一个名称对应的文件/目录移至对应目录下，若没有则当做是需要重命名。

当要移动至的文件夹中存在重名文件时，就出现了是否覆盖的问题，而默认是不提示直接覆盖。

```bash
mv -i <名称1> <名称2> #若出现重名，提醒是否覆盖，用y/n来确认
mv -f <名称1> <名称2> #覆盖已有文件上时不进行任何提示
mv -b <名称1> <名称2> #为将要被覆盖的文件创建一个备份
mv -u <名称1> <名称2> #当源文件比目标文件新（根据时间戳判断），或目标文件不存在时才执行移动操作
```

```bash
.
├── dir1
│   ├── file2.txt
│   ├── file.txt #移入并覆盖的文件
│   └── file.txt~ #被覆盖的文件备份，扩展名后带有'~'
└── dir4

2 directories, 3 files
```

### `cp`

```bash
cp <file1> <file2> #将文件1复制一份存在同一目录下，并命名为file2
cp -r <目录1> <目录2> #递归复制目录1及其下内容，并命名为目录2
cp -v <file1> <file2> #逐行输出复制的过程，便于检查
```

和`mv`一样，遇到重名默认直接覆盖

```bash
cp -i <file1> <file2> #若出现重名，提醒是否覆盖，用y/n来确认
cp -f <file1> <file2> #覆盖已有文件上时不进行任何提示
cp -b <file1> <file2> #为将要被覆盖的文件创建一个备份
```

复制时默认不会复制属性。需要手动加上`-a`

```bash
cp -a <file1> <file2> #连带属性一同复制
```

## 文件操作

### `stat`

```bash
stat <file> #显示文件信息
stat -f <file> #显示文件系统中的文件信息，如储存空间等
stat -t #以简洁的方式显示
```

> 对于文件属性中的3个时间戳：
>
> - `Access`/`atime`：访问时间，在修改文件或读取文件时改变
> - `Modify`/`mtime`：修改时间，在修改文件时改变
> - `Change`/`ctime`：状态改变时间，在修改文件或文件属性变化时改变

### `rename`

用字符串替换的方式批量改变file

```bash
rename 's/old/new/' <file> #两个参数都可以使用通配符
```

其中`old`为旧文件名中需要被替换的部分，`new`为用来更新的字符串

```bash
rename -n 's/old/new/' <file> #预览修改前后，防止重命名有误
rename -v 's/old/new/' <file> #逐步输出重命名过程
rename -f 's/old/new/' <file> #强制改写
```

## 查找

### `file`

win下可以通过文件后缀名来识别文件类型，但是在linux中，只能通过`file`来识别。

```bash
file <file>
file -b <file> #列出文件类型，不显示文件名称
file -c <file> #详细显示指令执行过程
file -f <file> #指定名称文件，显示多个文件类型信息
file -L <file> #直接显示某个符号链接指向的文件类别
file -m <file> #指定魔法数字文件
file -z <file> #尝试去解读压缩文件的内容
file -i <file> #显示MIME类别
```



### `find`

> ```bash
> find [dir] [param] [condition] #支持正则表达式
> ```

```bash
find <dir> -name <name> #匹配文件名
find <dir> -iname <name> #同上，但忽略大小写
find <dir> ! -iname <name> #搜索文件名不是给定值的文件

find <dir> -perm <mode> #(permission)匹配权限（完全匹配）
find <dir> -perm -<mode> #匹配权限（包含即可）

find <dir> -user <name> #匹配所有者
find <dir> -nouser <name> #匹配无所有者的文件
find <dir> -group <name> #匹配所有组
find <dir> -nogroup <name> #匹配无所有组的文件

find <dir> -mtime -n (+n) #匹配访问文件的时间（-n为n天以内，+n为n天以前），atime/ctime同理
find <dir> -newer f1 !f2 #匹配比f1新但比f2旧的文件

find <dir> -type b/d/c/p/l/f #匹配文件类型，后面的字符依次表示设备/目录/字符设备/管道/链接/文本文件
find <dir> -size +<size> (-<size>) #匹配文件大小（+50K为超过，-50K为不超过）

find <dir> -prune <name> #忽略某个目录

find <dir> -exec <command> {} \; #对搜索出来的文件执行指定命令
```

### `which`

用来查找系统命令的位置。当输入某个命令时，`which`会告诉你这个命令的实际路径。

**注意**：`which`只能查找系统命令，即只能在系统环境变量路径`$PATH`中找。当然，`$PATH`可以手动修改

```bash
which typora
```

输出

```bash
/snap/bin/typora
```

### `whereis`

用于查找**命令**的二进制文件、源代码和`man`手册页等相关文件的路径

```bash
whereis -b <command> #查找二进制程序或命令
whereis -B <dir> -f <command> #在指定目录查找二进制程序或命令

whereis -m <command> #查找man手册文件
whereis -M <dir> -f <command> #在指定目录查找man手册文件

whereis -s <command> #只查找源代码文件
whereis -S <dir> -f <command> #从指定目录下查找源代码文件
```

## 权限操作

### `chmod`

改变文件权限

符号模式：

```bash
chmod <ugoa> <+-=> <rwx> <file>
# u：user，文件所有者
# g：group，文件所有者所在组
# o：others，所有其他用户
# a：all，所有用户，相当于ugo
# +/-/=：增加/去除/直接设定
```

```bash
chmod -R a+r * #将当前目录下所有文件递归设置为所有人可读
```

数字模式：

<img src="/home/maoting/snap/typora/96/.config/Typora/typora-user-images/image-20250721192718610.png" alt="image-20250721192718610" style="zoom: 33%;" />

```bash
chmod <num> <file>
```

## 文本处理

### `grep`

文本搜索工具

```bash
grep -i <content> <file>#忽略大小写

grep -w <content> <file> #匹配整行
grep -x <content> <file> #匹配整词
grep -F <string> <file> #显示固定字符串的内容
grep -v <content> <file> #显示不包含匹配文本的所有行

grep -r <content> <dir> #递归搜索
grep -l <content> <file> #只显示含匹配内容的文件名，常与-r配合使用
grep -h <content> <file> #查询多文件时返回结果中不显示文件名

grep -E <content> <file> #支持扩展的正则表达式
```

### `less`

分页查看长文本

```bash
less <file> #用Pgup/Pgdn翻页，上下键滚动

less <file>
/<keyword> #逐个查找关键字（高亮显示）
           #按N查找下一个，按shift+N查找上一个
```



## 磁盘与文件系统

### `df`

disk free，显示磁盘空间占用情况

```bash
df -a #显示所有类型的文件系统的磁盘占用
df -h #以易于阅读的方式显示磁盘占用
df <param> <dir> #显示指定目录的磁盘占用
df -T #同时显示文件系统类型
df -t <type> #显示指定类型的文件系统占用情况
```

### `tar`

tape archive，磁盘存档，即打包解压文件，注意，打包不等于压缩

```bash
tar -c <newtar> #新建打包文件<newtar>，不能单独使用，因为没有指定要打包哪些文件，需要加上-f选项
tar -cf <newtar> <files> #将<files>打包为<newtar>，<files>可以是一个或并列的多个文件，也可以用通配符一次匹配多个文件
tar -cvf <newtar> <files> #同时显示操作过程

tar -zcvf <newzip> <files> #通过gzip的方式打包并压缩，压缩文件以tar.gz为后缀
tar -tf <newzip> #列出压缩包里的文件

tar -zxvf <newzip> #通过gzip的方式解压
tar -zxvf <newzip> -C <dir> #通过gzip的方式解压到<dir>
```

`tar`的功能有限，不能将已有的tar文件压缩为tar.gz文件

### `gzip/gunzip`

压缩率可以达到60%-70%，便于节省磁盘空间和网络传输

```bash
gzip <file> #直接将原文件压缩成.gz文件并覆盖原文件
gzip -k <file> #直接将原文件压缩成.gz文件并保留原文件

gzip -d <newzip> #解压缩并覆盖原压缩包（只能对.gz文件操作）
gunzip <newzip> #同上

gzip -t <newzip> #检查压缩包是否损坏，若没有内容返回就说明正常
```

### `zip/unzip`

```bash
zip <newzip> <files> #将<files>打包为<newzip>，<files>可以是一个或并列的多个文件，也可以用通配符一次匹配多个文件
zip -r <newzip> <dir> #递归压缩目录下的所有文件
zip -v #显示操作过程
zip -d <newzip> <file> #更新压缩包内的文件，将<file>文件添加到已有的压缩包<newzip>里
```

```bash
unzip -l <newzip> #不解压，仅查看压缩包中的内容
unzip -v <newzip> #查看压缩包的内容，以及压缩比率
unzip -t <newzip> #检查压缩包是否损坏
unzip <newzip> #解压到当前目录
unzip <newzip> -d <dir> #解压到指定目录
```

## 系统管理与性能监控

### `dmesg`

用于显示开机信息。由于这些信息开机时都在屏幕上一闪而过，用dmesg就可以将这些信息再次调出来查看

```bash
dmesg | less #分页查看信息，因为dmesg的信息量非常庞大
dmesg | grep -i tty #查看TTY设备，USB/DMA/memory同理
dmesg -x #同时显示信息级别（包括emerg/alert/crit/err/warn/noice/info/debug）
dmesg --level=err,warn #显示特定级别的信息，注意逗号后面不能有空格
dmesg -T #同时显示时间戳
dmesg -r #显示原始数据
```

### `free`

显示内存占用情况

```bash
free #以默认格式显示
free -h #以易读格式显示
free -hs 0.5 #以易读格式，持续显示内存占用，每0.5秒更新一次

free -b #以Byte为单位
free -k #以kb为单位
free -m #以mb为单位
```

## 网络工具

### `ping`

Packet Internet Groper的缩写，因特网包探索器，用于测试主机间的联通性

```bash
ping <目标主机> #一直ping
ping -c <count> <目标主机> #指定发送报文次数
ping -c <count> -i <interval> <目标主机 #以一定时间间隔（单位秒）发送指定次数的报文
```

### `wget`

网络下载，支持断点下载

```bash
wget <url> #下载指定url对应的文件
wget -i <file> #用于下载多个文件。把各个文件的url都列在<file>（通常是txt）里，然后在<file>中读取url下载
wget -O <name> <url> #下载文件后重命名
wget -P <dir> <url> #保存到指定文件夹
wget -c <url> #开启断点续传
wget -b <url> #开启后台下载
tail -f wget-log #后台下载时查看下载进度
```

## 进程管理

### `top`

实时显示进程动态

```bash
top #实时显示进程动态
top -c #显示完整的进程信息
top -d <sec> #指定刷新时间
top -p <pid> #只监控指定进程号的状态
top -n <num> #设置信息更新次数
```

> 快捷键：
>
> - `c` 显示进程的绝对路径
> - `P` 根据CPU使用率排序
> - `M` 根据物理内存使用率排序
> - `q` 退出

### `ps`

process states，常用参数的组合：

```bash
ps -aux | less #列出所有在内存中的程序
ps -A | less #显示所有进程的信息
ps -ef | less #显示所有进程的信息，连同命令行
ps -axf #树形显示所有进程
ps -aux | sort -rnk 3 | less #根据第三列降序
```

> 部分信息说明：
>
> - `VSZ`：占用虚拟内存的大小
> - `RSS`：占用物理内存的大小
> - `TTY`：运行在哪个终端上面，若与终端无关则不显示
> - `STAT`：进程状态
>   - `R`：运行
>   - `S`：可中断睡眠
>   - `D`：不可中断睡眠
>   - `T`：停止
>   - `Z`：僵死
> - `COMMAND`：该进程对应的执行程序

### `pstree`

以树状图显示进程

```bash
pstree -a #显示每个进程的完整指令，包括路径/参数等等
pstree -p #显示所有进程的进程号和id
```

### `pgrep`

检索进程

```bash
pgrep <name> -d' ' #将检索到的进程的id在同一行输出，用'  '分隔
pgrep -l <name> #输出进程id和进程名
pgrep -fl <name> #扩大搜索范围，在进程路径等信息中也进行搜寻
pgrep <regex> -l #用正则表达式匹配
```

### `lsof`

查看进程打开的文件

```bash
lsof #查看所有文件与进程对应的信息
lsof +d <dir> #查看指定目录中文件与进程对应的信息
lsof +D <dir> #递归查看指定目录所有文件与进程对应的信息
lsof <file> ##查看哪个进程与指定文件有关
lsof -c <precess> #查看指定进程打开的文件
lsof -p <pid> #查看指定i对应的进程打开的文件
```

### `kill`

发送信号到进程

```bash
kill -l #列出系统支持的所有信号列表
kill -l <SIG> #得到信号对应的序号
kill -<index> <pid> #对指定进程发送信号
kill -9 $(ps -ef | grep maoting) #杀死指定用户的所有进程
```

>常用信号：
>
>- `HUP 1`：终端断线
>- `INT 2`：中断（同`ctrl+c`）
>- `QUIT 3`：退出（同`ctrl+\`）
>- `TERM 15`：终止
>- `KILL 9`：强制终止
>- `STOP 19`：暂停（同`ctrl+z`）
>- `CONT 18`：继续（与`STOP`对应）

### `killall`

使用进程名称来杀死进程

```bash
killall -l #列出系统支持的所有信号列表
killall <name> #杀死指定名称的进程
killall -<index> <name> #杀死指定名称的进程
```

## 包管理

### `apt`

advanced package tool，是Debian的Linux发行版的包管理软件

```bash
sudo apt update #列出可更新的软件清单
sudo apt upgrade #升级可更新的软件包
sudo apt upgrade <package> #更新指定软件包

sudo apt install <package> #下载并安装软件包

sudo apt remove <package> #删除指定软件包和相关库文件，但保留配置文件
sudo apt purge <package> #彻底删除指定软件包和相关库文件，以及所有相关配置文件
sudo apt autoremove #清理不再使用的依赖和库文件

sudo apt show <package> #显示指定软件包信息

sudo apt list --installed #显示已安装的软件包
sudo apt list <name>* #用通配符搜索软件包
```

### `snap`

 Ubuntu 系统中用于管理 Snap 包的命令行工具

```bash
sudo snap install <package> #安装
sudo snap remove <package> #卸载

sudo snap refresh #手动检查并更新所有已安装的 Snap 包
sudo snap refresh <package> #更新指定的 Snap 包

snap list #列出已安装的 Snap 包
snap find <package> #查找
snap info <package> #查看 Snap 包的详细信息
```

> **`snap` or `apt`？**
>
> Snap 包的出现主要是为了解决传统 Linux 软件包管理的一些痛点：
>
> - 依赖冲突： 传统软件包经常因为依赖库版本不兼容而导致问题。Snap 包将所有依赖都打包进去，避免了这种冲突。
> - 应用程序隔离： Snap 应用在沙盒环境中运行，与其他系统组件隔离，这增强了系统的安全性和稳定性。
> - 跨发行版兼容性： Snap 包可以在任何支持 Snap 的 Linux 发行版上运行，而不仅仅是 Ubuntu。
> - 自动更新： Snap 应用可以自动更新到最新版本，无需用户手动干预。
> - 更快的发布周期： 开发者可以更快地发布新版本应用，并直接推送给用户。
>
> `snap` 和 `apt` 都用于管理软件包，但它们之间存在根本区别：
>
> - `apt`： 管理传统的 `.deb` 包。这些软件包通常安装在系统级目录中，并与其他软件包共享系统库。`apt` 依赖于软件包仓库，并处理复杂的依赖关系。它更像是操作系统的“原生”包管理器。
> - `snap`： 管理 Snap 包。每个 Snap 包都是一个独立的、自包含的单元，在隔离的沙盒中运行。它们不依赖于系统中的其他库，从而避免了依赖冲突。Snap 包通常安装在 `/snap` 目录下。
>
> 什么时候用哪个？
>
> - 如果你想要稳定、经过发行版测试的软件包，或者你需要系统级别的集成（例如桌面环境组件），通常会使用 `apt`。
> - 如果你想要最新版本的应用程序，不希望遇到依赖问题，或者需要隔离和安全性，那么 `snap` 是一个很好的选择。许多流行的应用程序（如 Spotify, VS Code, Slack 等）都提供了 Snap 版本。

## 其他

### `source`

在当前shell环境中，从指定文件读取和执行命令，通常用于重新执行刚刚更新的文件并让其生效，也通常用命令`.`来代替

```bash
source <file>
```

### `echo`

在终端输出指定的字符串，或者变量提取后的值

```bash
echo <variable_name> #输出变量（例如$PATH）的值
echo \<variable_name> #取消转义
echo <name> > <file> #将输出结果重定向到文件
echo `<command>` #输出命令的执行结果
echo -e <string> #开启识别字符串中的转义符（默认不开启，直接当作普通字符输出）
echo -E <string> #禁止识别字符串中的转义符，与-e相反
```

### `bc`

linux自带的计算器

```bash
bc #进入计算器
quit #退出

+-*/ #四则运算
10^10 #指数运算（指数部分必须是整数）
sqrt(100) #开平方

e(2) #计算e^n
l(2) #计算ln(n)
echo "obase=10;ibase=2;110101" | bc #进制转换
```

### `ln`

为文件创建快捷方式（link）

```bash
ln <sourcefile> <hardfile> #为源文件创建硬链接
ln -s <sourcefile> <softfile> #为源文件创建软链接
```

>在 Linux 中，硬链接（Hard Link） 和 软链接（Soft Link），也称为符号链接（Symbolic Link），是两种不同的文件引用方式。
>
>硬链接就像一个文件的“别名”，它和原始文件共享相同的 inode 号。这意味着它们都指向硬盘上同一块物理数据。删除其中一个链接，只要还有其他链接存在，数据就不会丢失。你不能对目录创建硬链接，也不能跨文件系统创建。
>
>软链接则更像一个文件的“快捷方式”。它是一个独立的文件，有自己的 inode 号，但其内容是指向另一个文件的路径。如果原始文件被删除，软链接就会失效，变成“死链接”。软链接可以链接到目录，也可以跨文件系统使用。

## Conda

### 环境管理

```bash
conda env list #查看现有环境列表

conda create -n <env_name> python=x.x #创建新环境
conda create -n <new_env> --clone <source_env> #克隆一个已存在的环境
conda remove -n <env_name> --all #删除一个指定名称的环境及其所有包

conda activate <env_name> #激活（进入）一个已存在的环境
conda deactivate #关闭（退出）当前所在的环境，返回 base 环境

conda env export > environment.yml #导出当前环境的配置到 YAML 文件
conda env create -f environment.yml #从 YAML 文件创建新环境
```

### 包管理

```bash
conda install <package_name> #安装一个或多个包
conda install <package_name>=x.x #安装指定版本的包

conda uninstall <package_name> #卸载一个或多个包
conda list #列出当前环境中已安装的所有包

conda search <package_name> #搜索可用的包版本

conda update <package_name>	#更
新指定的包
conda update --all #更新当前环境中所有可更新的包

conda clean -a #清理未使用的包和缓存（可以释放大量磁盘空间）
```

