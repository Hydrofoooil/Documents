### 启动ROS

启动ros系统核心：

```bash
roscore
```

### `package`包

进入工作空间：

```bash
cd catkin_ws/src/
```

创建包：

```bash
catkin_create_pkg <包名> <依赖项列表>
```

> **依赖项：**有些“通用节点”或“通用资源”，绝大部分其他节点工作时都会和他们产生关联，因此这些“通用节点”或资源就成了一个通用的依赖项
>
> **包一般储存在两个位置：**
>
> - `/opt/ros/<ros版本号>/share`：里面的包是无需编译直接执行的可执行文件，他们有两个来源：
>
>   `sudo apt-get install ros-<ros版本>-desktop-full`安装时所带来的基础包
>
>   `sudo apt-get install ros-<ros版本>-XXX`之后另外下载的独立扩展包
>
> - `~/catkin_ws/src/`：自定义的软件包，是源码包，需要编译后执行

创建节点的源码文件：

```bash
touch <文件名>
```

直接跳转到对应的软件包地址：

```bash
roscd
```

显示子目录：

```bash
ls
```

### 编译

先回到工作空间目录，然后再编译

```bash
cd catkin_ws
catkin_make
```

### 运行

把工作空间的环境参数加载到终端里：

```bash
source ~/catkin_ws/devel/setup.bash
```

> 当运行 `catkin_make` 或 `catkin build` 之后，**ROS 会在 `devel/` 目录下自动生成或更新 `setup.bash` 文件**，里面是一些 **自动生成的 shell 脚本**，其作用是：
>
> - 设置环境变量（如 `ROS_PACKAGE_PATH`、`CMAKE_PREFIX_PATH`、`PYTHONPATH` 等）
> - 确保你新编译的包可以被 `rosrun`、`roslaunch`、`roscd` 等 ROS 工具识别
>
> 因此上述语句的作用：重新加载自己编译的 ROS 工作空间的环境设置，使得终端能够识别工作空间中的 ROS 包。

直接运行某个包里的一个节点：

```bash
rosrun <包名> <节点名>
```

### **`launch`文件**

launch文件是`.launch`后缀的XML格式文件，一次启动多个节点。

基本格式：

```xml
<launch>
	<!-- 启动一个 C++ 节点 -->
	<node name="robot_node" pkg="my_robot_pkg" type="my_node" output="screen" />

	<!-- 启动一个 Python 脚本节点 -->
	<node name="controller" pkg="my_robot_pkg" type="control.py" output="screen">
		<param name="speed" value="1.5" />
	</node>

	<!-- 设置参数 -->
	<param name="robot_name" value="zeta1" />

	<!-- 命名空间 -->
	<group ns="robot1">
		<node name="sensor" pkg="my_robot_pkg" type="sensor_node" output="screen" />
	</group>
</launch>

```

