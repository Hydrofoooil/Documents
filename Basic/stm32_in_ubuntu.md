## STM32连接Ubuntu：

### 获取USB转串口连接电脑的端口号：

使用`dmesg`显示内核识别信息：

```bash
sudo dmesg
```

会显示一大串内核信息，需要从中提取连接串口模块（e.g.CH341）的端口的信息：

```bash
sudo dmesg | grep -i ch341
```

会看到类似输出：

```vb
[    3.208888] usb 1-1: ch341-uart converter now attached to ttyUSB0
[    3.208900] usbcore: registered new interface driver ch341
```

表示ch341-uart converter（转换器）已经绑定在了`/dev/ttyUSB0`上

> **`dmesg`输出内容包含：**
>
> - 启动过程的硬件检测（如 CPU、内存、硬盘、USB 设备）
> - 驱动加载情况
> - 文件系统挂载
> - 网络接口初始化
> - 内核错误（如 OOM、kernel panic 等）
>
> **`grep`基本用法：**
>
> ```bash
> grep [选项] '搜索字符串' 文件名
> ```
>
> | 选项                                     | 说明                                         |
> | ---------------------------------------- | -------------------------------------------- |
> | `-i` (ignore)                            | 忽略大小写                                   |
> | `-r` 或 `-R` (recursion)                 | 递归搜索当前目录及子目录                     |
> | `-n` (number)                            | 显示匹配行的行号                             |
> | `-v` (invert match)                      | 显示不匹配的行                               |
> | `-l` (`--files-with-matches`)            | 只列出包含匹配信息的文件名                   |
> | `-c` (column)                            | 显示匹配的行数                               |
> | `-o` (`-only-matching`)                  | 只输出文件中匹配（通常指与正则表达式）的部分 |
> | `--color`                                | 高亮显示匹配部分                             |
> | `-E` (Extended Regular Expressions, ERE) | 启用正则表达式                               |
>
> 例如在`file.txt`中搜索`'hello'`这个单词，忽略大小写：
>
> ```bash
> grep  -i 'hello' file.txt
> ```
>
> 使用正则表达式：
>
> ```bash
> grep -E "error|fail|critical" file.txt
> ```
>
> 搭配管道使用：
>
> ```bash
> dmesg | grep usb
> ```
> 或者
> ```bash
> dmesg
> !! | grep usb
> ```
>
> > 在 Shell 里，`|` 符号称为“管道”（pipe），作用是把 **左边命令的输出结果**，当作 **右边命令的输入**。
>>
> > 其中`!!`代表引用上一个命令输出的结果
> >
> > 即先执行 `dmesg`（打印内核日志），再用 `grep` 找出含有 `usb` 的行

### **授权连接串口：**

python节点程序默认没有权限连接串口，不能烧录和接收，需要增加权限：

临时授权：

```bash
sudo chmod 666 /dev/ttyUSB0
```

> 注意：这个权限在重启或重新插拔设备后会失效。

永久授权：

```bash
sudo usermod -a -G dialout $USER
```

> 然后 **注销并重新登录（或重启系统）** 让权限生效。
>
> 该语句的作用是把当前用户加入串口权限组 `dialout`

### 烧录：

用` stm32flash`工具

注意烧录前32的板子要进入烧录模式（bootloader模式）

```bash
sudo stm32flash -w XXX.hex -v -g 0x0 /dev/ttyUSB0
```

`-w`：（write）写入hex文件

`-v`：（verify）写入后验证

`-g 0x0`：烧录后跳转到用户程序

`/dev/ttyUSB0`：USB转串口连接的端口

### 串口读数：

连接USB转串口

执行：（假设链接端口的设备名是`/dev/ttyUSB0`）

```bash
ls /dev/ttyUSB* /dev/ttyUSB0 
```

> `/dev/ttyUSB0` 是 Linux 系统中的一个**设备文件**，表示一个通过 **USB 转串口芯片（如 CH340、CP2102、FT232 等）** 接入系统的串口设备。
>
> | 部分    | 含义                                   |
> | ------- | -------------------------------------- |
> | `/dev/` | 所有设备文件所在目录                   |
> | `tty`   | 表示“终端”（teletype），用于串口类通信 |
> | `USB0`  | 第一个 USB 串口设备（0 号）            |

打开minicom：

```bash
minicom -b 115200 -D /dev/ttyUSB0
```

会进入空白窗口，然后会显示单片机通过串口发送的信息

退出：Ctrl+A，然后按X

> **注意：**在ubuntu中，`\n`代表将光标移到下一行，但是不会移回下一行的最前面，只是向下平移。而`\r`才是将光标移回最前面。所以windows中的`\n`在ubuntu中需要`\r\n`才能实现。