## Docker初步

### Docker是什么？

**Docker 是一个“容器引擎”**，用于将软件及其所有依赖打包成一个“容器”（container），使得程序能**在任何地方一致运行**，包括不同的操作系统或服务器上。

- 容器 ≈ “轻量级虚拟机”
- 容器包括：代码 + 系统库 + 运行环境
- **一次构建，处处运行**

### 基本概念

| 名称                | 解释               |
| ----------------- | ---------------- |
| **Image（镜像）**     | 容器的模板（类似 ISO）    |
| **Container（容器）** | 镜像运行出来的实例        |
| **Dockerfile**    | 用来构建镜像的脚本        |
| **Volume（卷）**     | 持久化数据（容器删了数据还在）  |
| **Docker Hub**    | 官方镜像仓库，类似 GitHub |

一个镜像可以构建多个容器，镜像是静态的模板，容器是动态的实例，容器可以被创建、运行、停止和删除。

### 常用Docker命令

#### 拉取镜像

**从远程镜像仓库（默认 Docker Hub）下载一个镜像到你本地的 Docker 环境中**：

```bash
docker pull <镜像名>:<标签>
```

> 标签（`tag`）即为版本，`:<标签>`可省略，省略则默认下载最新版本（`defaut tag: latest`）
>
> 下载后的镜像会存在 `/var/lib/docker`目录

#### 查看本地镜像：

```bash
docker images
```

#### 删除镜像：

```bash
docker rmi <镜像ID 或 镜像名称>
```

> | 命令举例                    | 删除对象                                                   | 使用方式       | 优点             | 缺点                     |
> | --------------------------- | ---------------------------------------------------------- | -------------- | ---------------- | ------------------------ |
> | `docker rmi 2f1d13d03e78` | **通过镜像 ID 删除镜像**                             | 精准、不会歧义 | 精确删除指定版本 | 不容易记住，ID 会变      |
> | `docker rmi ubuntu`       | **通过镜像名称删除镜像**（实际是 `ubuntu:latest`） | 更直观、易读   | 方便记忆         | 可能有多个标签、版本冲突 |

#### 创建并运行新容器：

```bash
docker run <选项> <镜像名>
```

> | 选项       | 功能               |
> | ---------- | ------------------ |
> | `-it`    | 交互式             |
> | `-d`     | 后台运行           |
> | `--name` | 容器命名           |
> | `-p`     | 端口映射           |
> | `-v`     | 挂载卷             |
> | `rm`     | 容器退出后自动删除 |

#### 创建并以交互式模式运行容器：

```bash
docker run -it ubuntu
```

> | 选项   | 说明                       |
> | ------ | -------------------------- |
> | `-i` | 交互式运行（保持输入）     |
> | `-t` | 分配一个伪终端（terminal） |
>
> 然后会进入一个 Ubuntu shell：
>
> ```ruby
> root@a2d3f4g5h6:/#
> ```
>
> 就相当于一个ubuntu虚拟机，可以在里面进行各种ubuntu的操作

#### 在创建时给容器命名：

```bash
docker run --name <容器名> <镜像名>
```

> 如果创建时不给容器命名，Docker会自动给容器起一个“形容词_人名”格式的随机名字，例如 `stoic_leavitt`，可以之后手动重命名
>
> `--name`和 `-it`可以一起使用

#### 给容器重命名：

```bash
docker rename <旧容器名或ID> <新容器名>
```

> **为了简便，ID可以只输前四位，下同**

#### 查看运行中的容器：

```bash
docker ps
```

#### 查看所有容器（运行的和停止的）：

```bash
docker ps -a
```

#### 启动一个已有的容器：

```bash
docker start -i <容器名>
```

#### 强制启动容器：

```bash
docker restart <容器名>
```

> | 命令               | 行为                                                                     |
> | ------------------ | ------------------------------------------------------------------------ |
> | `docker start`   | **启动一个已停止的容器**，不重启运行中的                           |
> | `docker restart` | **无论是否在运行，都强制重启容器**（对于运行中的先 stop 再 start） |

#### 停止容器：

```bash
docker stop <容器名>
```

#### 删除容器：

```bash
docker rm 容器名
```

> 注意和删除镜像的 `docker rmi`区别开

非交互式启动后手动进入/退出容器（两种方法）：

```bash
docker exec -it <容器名 或 ID> <要在容器内启动的可执行程序>
```

> 通常为 `docker exec -it /bin/bash`， 其中 `/bin/bash`为ubuntu里在终端输入命令、运行脚本时的解释器
>
> 运行后会在容器中新建一个终端来接受指令
>
> Ctrl+D 或输入 `exit`退出，由于是新建终端，退出的只是这个终端，容器不会关闭，需要另外 `stop`来停止容器

```bash
docker attach <容器名 或 ID>
```

> 运行后会直接为我们**接入容器主进程的已有终端**，之后就在此输入、运行命令行，而非新建终端来输入、运行命令行
>
> Ctrl+C 或输入 `exit`退出，会连同容器一起

显示当前位置的绝对路径：

```bash
pwd
```

在容器和主机之间相互复制文件：

```bash
docker cp <主机上的源路径> <容器名称或ID>:<容器内的目标路径>
docker cp <容器名称或ID>:<容器内的源路径> <主机上的目标路径>
```

> 如果显示mkdir 的permission denied 的话就加sudo

#### 将容器保存为镜像：

```bash
docker commit my_ros my_custom_ros_image
```



### 数据卷挂载

用于在Docker容器里运行或访问主机（Host）上的文件。

**注意：不能为一个已经创建好的容器动态添加或修改挂载目录，只有在创建容器时可以设置挂载目录**

相当于在主机和容器之间建立一个共享文件夹，容器内的某个路径会直接映射到主机上的一个文件或目录。你在任何一边对这个共享文件夹里的内容进行修改，另一边都能立刻看到变化。

```bash
docker run -it -v <主机上的目录>:<容器上的目录> <image_name>
#或者：
docker run -it --mount type=bind,source=<主机上的目录>,target=<容器上的目录> <image_name>
```

查看当前容器挂载了哪些目录：

```bash
docker inspect --format='{{range .Mounts}}Source: {{.Source}}{{"\n"}}Destination: {{.Destination}}{{"\n"}}---{{end}}' <容器名称或ID>  #位于容器外查看
```

```bash
findmnt  #位于容器内查看
```

| **特性**     | **docker cp (复制)**                                         | **数据卷挂载 (-v 或 --mount)**                               |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **用途**     | **一次性**的文件传输。                                       | **持久化**和**实时同步**。                                   |
| **典型场景** | 1.  向正在运行的容器注入一个配置文件。 2.  从容器中取出日志文件或生成的结果。 3.  进行快速调试。 | 1.  开发环境，主机代码修改后容器立即生效。 2.  运行数据库，将数据目录持久化在主机上。 3.  共享配置文件或大型数据集。 |
| **数据关系** | 复制后，主机和容器内的文件**相互独立**，修改一方不影响另一方。 | 主机和容器共享同一个文件/目录，任何一方的修改都**实时可见**。 |
| **操作时机** | 可以在容器**运行期间**随时执行。                             | 必须在 `docker run` **创建容器时**就定义好。                 |

### 在docker里启动GUI程序

标准的Docker容器是一个隔离的、“无头”（headless）的命令行环境。它内部没有图形界面，也不知道宿主机（您的Ubuntu桌面）有显示器。

#### 本地docker启用GUI

若尝试启动一个图形用户界面（GUI）应用程序例如 `RViz`，但它无法找到或连接到任何可用的“屏幕”或“显示器”，就会报错。

**第1步：**在Ubuntu主机上（不是在容器里！）打开一个新终端，运行以下命令，授权来自本地的连接：

```bash
xhost +local: #授权来自本地的连接
xhost +inet:192.168.1.100 #举例：授权来自某个ip的连接
xhost + #允许任何主机的任何应用程序连接到我的 X Server 并在我的屏幕上显示内容。
```

  **第2步：**使用带有特定参数的 `docker run` 命令来启动您的容器。

  您需要在 `docker run` 命令中加入以下两个关键参数：

- `-e DISPLAY=$DISPLAY`：将您主机的 `DISPLAY` 环境变量传递给容器，告诉容器去哪里找显示器。

- `-v /tmp/.X11-unix:/tmp/.X11-unix`：将X11的通信套接字（socket）文件挂载到容器内部，为容器和主机建立一条图形通信的“管道”。

-  一个完整的示例 `docker run` 命令可能如下所示：

```bash
docker run -it \
	--net=host \ #建议加上。这样 Docker 会直接使用宿主机的网络，否则你的 ROS 节点可能无法和外部（比如你真机上的硬件或其他节点）通信。
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v <主机上的目录>:<容器上的目录> \
    <你的镜像名称>
```

  **如果您已经有一个正在运行的容器**，您需要停止它，然后使用 `docker start` 和 `docker exec` 是不够的，必须在 `run` 的时候就添加这些参数。

 **第3步：**在配置好的容器内部，再次运行 `roslaunch` 命令。 RViz现在应该可以正常启动了。

#### SSH启用GUI

又或者在通过SSH远程连接到另一台机器（比如您的树莓派或服务器）上运行此命令时，由于标准的SSH连接只传输命令行文本，因此在远程机器上启动RViz时，它会尝试在那台远程机器上寻找一个显示器，但通常那里并没有。它不知道应该把图形窗口显示回你自己电脑的屏幕上。

您需要告诉SSH在连接时，也一并建立一条图形信息的“隧道”。

**操作方法非常简单**：在您发起SSH连接时，添加一个 `-X` 或 `-Y` 参数。

```bash
ssh -X <用户名>@<远程机器的IP地址>
```

**`-X` 和 `-Y` 的区别**：`-X` 是标准的X11转发，安全性更高。`-Y` 是“可信的”X11转发，限制更少，在某些应用（特别是使用3D加速的，如RViz）上可能兼容性更好。如果 `-X` 不起作用，可以尝试 `-Y`。

使用这种方式登录后，您在远程终端里启动任何GUI程序（如 `rviz` 或 `gedit`），它的窗口都会自动出现在您本地的Ubuntu桌面上。

### 日常使用

应该采用“创建一个容器，然后一直复用它”的模式。

开启容器：

```bash
docker start my_ros
docker exec -it my_ros bash
```

关闭容器：

```bash
exit
docker stop my_ros
```

到一定阶段将容器保存为新的镜像作为存档

```bash
docker system df -v
```

查看镜像、容器的状态、占用以及引用关系，最全面



## docker compose

 **Docker Compose（v2：`docker compose`）常用命令速查**。默认你在有 `compose.yml` 的目录里；如果用指定文件，就在命令里加 `-f path/to/compose.yml`。

------

### 启动与停止

#### 启动（前台）

```bash
docker compose up
```

#### 启动（后台）

```bash
docker compose up -d
```

#### 启动并强制构建镜像

```bash
docker compose up -d --build
```

#### 仅启动某个服务

```bash
docker compose up -d openpi_server
```

#### 停止（保留容器）

```bash
docker compose stop
```

#### 停止并删除容器/网络（不删你宿主机目录；会删 named volumes 需谨慎）

```bash
docker compose down
```

#### down 并删除 volumes（⚠️ 会删 named volume，可能导致数据丢失）

```bash
docker compose down -v
```

------

### 日志与“重新 attach”

#### 跟随所有服务日志（最常用）

```bash
docker compose logs -f
docker compose -f examples/libero/compose.yml logs -f openpi_server
```

#### 只看某个服务日志

```bash
docker compose logs -f openpi_server
```

#### 只看最后 N 行

```bash
docker compose logs --tail 200 openpi_server
```

------

### 查看状态与信息

#### 看服务/容器状态

```bash
docker compose ps
```

#### 展开后的最终配置（变量替换后，排查 yml 很好用）

```bash
docker compose config
```

------

### 进入容器与执行命令

#### 在**运行中的**服务里执行命令（推荐）

```bash
docker compose exec openpi_server bash
```

#### 直接执行一条命令

```bash
docker compose exec openpi_server nvidia-smi
```

#### 启一个“临时容器”跑命令，命令结束就删（不影响现有容器）

```bash
docker compose -f examples/libero/compose.yml run openpi_server bash -lc 'uv run scripts/serve_policy.py --help | sed -n "1,200p"'
```

------

### 镜像构建与拉取

#### 单独构建镜像

```bash
docker compose build
```

#### 只构建某个服务

```bash
docker compose build openpi_server
```

#### 不用缓存构建（很慢，但排查依赖问题有用）

```bash
docker compose build --no-cache
```

#### 拉取镜像（对 image: 指定的镜像有效）

```bash
docker compose pull
```

------

### 重启与更新

#### 重启服务（不重建容器）

```bash
docker compose restart openpi_server
```

#### 重新创建容器（配置变了常用）

```bash
docker compose up -d --force-recreate openpi_server
```

#### 重新创建并重新 build

```bash
docker compose up -d --build --force-recreate openpi_server
```

------

### 清理（谨慎）

#### 删除“已停止”的容器、无用网络等（全局）

```bash
docker system prune
```

#### 连镜像也删（⚠️更激进）

```bash
docker system prune -a
```

