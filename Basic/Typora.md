# markdown的使用说明

## 一、标题

>语法：# (一级标题)  ## (二级标题)  ### (三级标题) ......

>代码：
>
>```text
># 这是一级标题
>## 这是二级标题
>```

>效果:  
>
># 这是一级标题
>
>## 这是二级标题

>快捷键:
>
>* Ctrl+数字1~6可以快速将选中的文本调成对应级别的标题
>* Ctrl+0可以快速将选中的文本调成普通文本
>* Ctrl+加号/减号对标题级别进行加减

## 二、段落

### 1、换行


>代码:  
>```text
>这是一个段落
>这是一个段落
>```

>效果: 
>
>这是一个段落
>这是一个段落

### 2、分割线
>语法:  ---或者***+回车

>代码:
>```text
>---或者***
>```

>效果:
>
>---

## 三、文字显示

### 1、字体

>语法:
>
>* 粗体:  用一对双星号包裹
>* 删除线:  用一对双飘号包裹
>* 下划线:  用一对u标签包裹
>* 斜体:  用一对单星号包裹
>* 高亮:  用一对双等号包裹

>代码:
>```text
>**这是粗体**
>~~这是删除线~~
><u>这是下划线</u>
>*这是斜体*
>==这是高亮==
>```

>效果:
>**这是粗体**
>~~这是删除线~~
><u>这是下划线</u>
>*这是斜体*
>==这是高亮==

>快捷键:
>* 加粗:  Ctrl+B
>* 删除线:  Shift+Alt+5
>* 下划线:  Ctrl+U
>* 斜体:  Ctrl+I

### 2、上下标
>代码:
>```text
>x^2^
>H~2~O
>```

>效果:
>x^2^
>H~2~O

## 四、列表
### 1、无序列表
>代码:
>```text
>*/-/+ +空格
>```

>效果:
>1.只有同一级别:
>
>* 苹果
>* 香蕉
>* 橘子
>
>2.子集类:
>
>* 一级分类
>   * 二级分类 
>		* 三级分类

>快捷键:  Ctrl+Shift+]

### 2、有序列表
>代码:
>```text
>数字+.+空格
>```

>效果:
>1. 第一个标题
>2. 第二个标题
>3. 第三个标题
>
>	  * 子内容1
>	  * 子内容2
>4. 第四个标题

>快捷键:  Ctrl+Shift+[

### 3、任务列表
>代码:
>```text
>- [ ] 吃早餐
>- [x] 背单词
>```

>效果:
>- [ ] 吃早餐
>- [x] 背单词

## 五、区块显示
>代码:
>```text
>>+回车
>```

>效果:
>>这是最外层区块
>>
>>>这是内层区块
>
>>>>这是最内层区块

## 六、代码显示
### 1、行内代码
>代码:
>```text
>`int a=0;`（说明：`位于Esc下面）
>```

>效果:
>`int a=0;`

>快捷键:  Ctrl+Shift+`

### 2、代码块
>代码:
>
>````text
>```js/java/c#/text
>内容
>```
>````
>
>````

>快捷键:  Ctrl+Shift+K

## 七、链接
>代码:
>
>```text
>www.baidu.com
>[百度一下](https://www.baidu.com)
>[百度一下](https://www.baidu.com "https://www.baidu.com")
>```

>效果:
>www.baidu.com
>[百度一下](https://www.baidu.com)
>[百度一下](https://www.baidu.com "https://www.baidu.com")

>快捷键:  Ctrl+K

## 八、脚注

>说明:  对文本进行解释说明。

>代码: 
>
>```text
>[^文本]
>[^文本]:解释说明
>```

>效果:
>这是一个技术[^①]
>
>[^①]: 这是一个非常好用的框架。

## 九、图片插入
>代码:
>```text
>![不显示的文字](图片路径 "图片标题")
>```

>效果:
>![This is a picture](C:\Users\asus\Pictures\Saved Pictures\Snipaste_2020-09-03_13-19-11.png "漫步图")
>
>(注：效果路径为C:\Users\asus\Pictures\Saved Pictures\Snipaste_2020-09-03_13-19-11.png。在其他电脑上可能不显示。)

>快捷键:  Ctrl+Shift+I

## 十、表格

>代码:
>```text
>|  1   |  2   |  3   |
>| :--- | :--: | ---: |
>|  4   |  5   |  6   |
>|  7   |  8   |  9   |
>|  10  |  11  |  12  |
>```

>效果:
>
>| 1    |  2   |    3 |
>| ---- | :--: | ---: |
>| 4    |  5   |    6 |
>| 7    |  8   |    9 |
>| 10   |  11  |   12 |

>快捷键:  Ctrl+T

## 十一、流程图

### 1、横向流程图

> 代码:
>
> ````text
> ```mermaid
> graph LR
> A[方形]==>B(圆角)
> B==>C{条件a}
> C-->|a=1|D[结果1]
> C-->|a=2|E[结果2]
> F[横向流程图]
> ```
> ````
> 
> ````

>效果:
>```mermaid
>graph LR
>A[方形]==>B(圆角)
>B==>C{条件a}
>C-->|a=1|D[结果1]
>C-->|a=2|E[结果2]
>F[横向流程图]
>```

### 2、竖向流程图

> 代码:
> ````text
> ```mermaid
> graph TD
> A[方形]==>B(圆角)
> B==>C{条件a}
> C-->|a=1|D[结果1]
> C-->|a=2|E[结果2]
> F[竖向流程图]
> ```
> ````
> 
> ````

>效果:
>```mermaid
>graph TD
>A[方形]==>B(圆角)
>B==>C{条件a}
>C-->|a=1|D[结果1]
>C-->|a=2|E[结果2]
>F[竖向流程图]
>```

## 十二、表情符号

>代码:
>```text
>:happy:、:cry:、:man:
>```

>效果:
>:happy:、 :cry:、 :man:

## 十三、数学公式的输入

### 1、公式的插入

#### ①行中公式

>代码:
>```text
>$公式$
>```

>效果:
>$公式$

#### ②独立公式

>代码:
>
>```text
>$$
>公式
>$$
>```

>效果:
>$$
>公式
>$$

### 2、上下标
>代码:
>
>```text
>$x^{y^z}=(1+e^x)^{-2xy^w}$
>$\sideset{^1_2}{^3_4}{\underset{6}\bigotimes}$
>```

>效果:
>$x^{y^z}=(1+e^x)^{-2xy^w}$
>$\sideset{^1_2}{^3_4}{\underset{6}\bigotimes}$

### 3、括号和分隔符

>代码:
>```text
>$\langle\quad\rangle\quad\lceil\quad\rceil\quad\lfloor\quad\rfloor\quad\lbrace\quad\rbrace\quad\lVert\quad\rVert$
>$f(x,y,z)=3y^2z\left(3+\dfrac{7x+5}{1+y^2}\right)$
>$\left.\dfrac{\mathrm{d}u}{\mathrm{d}x}\right|_{x=0}$
>```

>效果:
>$\langle\quad\rangle\quad\lceil\quad\rceil\quad\lfloor\quad\rfloor\quad\lbrace\quad\rbrace\quad\lVert\quad\rVert$
>$f(x,y,z)=3y^2z\left(3+\dfrac{7x+5}{1+y^2}\right)$
>$\left.\dfrac{\mathrm{d}u}{\mathrm{d}x}\right|_{x=0}$

### 4、分数

>代码:
>
>```text
>$\frac{a}{b}  \quad\dfrac{a}{b}  \quad{a\over b}$
>```

>效果:
>$\frac{a}{b} \quad\dfrac{a}{b}  \quad{a\over b}$

### 5、开方

>代码:
>
>```text
>$\sqrt[根指数,省略时为2]{被开方数}$
>```

>效果:
>$\sqrt{2}\quad\sqrt[3]{2}$

### 6、省略号

>代码:
>```text
>$\cdots\quad\ldots\quad\vdots\quad\ddots$
>```

>效果:
>$\cdots\quad\ldots\quad\vdots\quad\ddots$

### 7、矢量和均值

>代码:
>
>```text
>$\overrightarrow{E(\vec{r})}\quad\overleftarrow{E(\vec{r})}\quad\overleftrightarrow{E(\vec{r})}\quad\underrightarrow{E(\vec{r})}\quad\underleftarrow{E(\vec{r})}\quad\underleftrightarrow{E(\vec{r})}\quad\overline{v}=\bar{v}\quad\underline{v}$
>```

>效果:
>$\overrightarrow{E(\vec{r})}\quad\overleftarrow{E(\vec{r})}\quad\overleftrightarrow{E(\vec{r})}\quad\underrightarrow{E(\vec{r})}\quad\underleftarrow{E(\vec{r})}\quad\underleftrightarrow{E(\vec{r})}\quad\overline{v}=\bar{v}\quad\underline{v}$

### 8、积分

>代码:
>
>```text
>$$
>\iint\limits_D\left(\dfrac{\partial Q}{\partial x}-\dfrac{\partial P}{\partial y}\right){\rm d}x{\rm d}y=\oint\limits_LP{\rm d}x+Q{\rm d}y
>$$
>```

>效果:
>$$
>\iint\limits_D\left(\dfrac{\partial Q}{\partial x}-\dfrac{\partial P}{\partial y}\right){\rm d}x{\rm d}y=\oint\limits_LP{\rm d}x+Q{\rm d}y
>$$

### 9、极限

>代码:
>
>```text
>$\lim\limits_{n\to\infin}(1+\dfrac{1}{n})^n=e$
>```

>效果:
>$\lim\limits_{n\to\infin}(1+\dfrac{1}{n})^n=e$

### 10、累加、累乘及交集、并集

>```text
>$\sum\limits_{i=1}^n\dfrac{1}{n^2}\quad and\quad\prod\limits_{i=1}^n\dfrac{1}{n^2}\quad and\quad\bigcup\limits_{i=1}^n\dfrac{1}{n^2}\quad and\quad\bigcap\limits_{i=1}^n\dfrac{1}{n^2}$
>```

>效果:
>$\sum\limits_{i=1}^n\dfrac{1}{n^2}\quad and\quad\prod\limits_{i=1}^n\dfrac{1}{n^2}\quad and\quad\bigcup\limits_{i=1}^n\dfrac{1}{n^2}\quad and\quad\bigcap\limits_{i=1}^n\dfrac{1}{n^2}$

### 11、希腊字母

| 语法                          | 字母                            | 语法                    | 字母                      | 语法               | 字母                 |
| ----------------------------- | ------------------------------- | ----------------------- | ------------------------- | ------------------ | -------------------- |
| \Alpha(\alpha)                | $\Alpha(\alpha)$                | \Beta(\beta)            | $\Beta(\beta)$            | \Gamma(\gamma)     | $\Gamma(\gamma)$     |
| \Epsilon(\epsilon)\varepsilon | $\Epsilon(\epsilon)\varepsilon$ | \Zeta(\zeta)            | $\Zeta(\zeta)$            | \Eta(\eta)         | $\Eta(\eta)$         |
| \Iota(\iota)                  | $\Iota(\iota)$                  | \Kappa(\kappa)\varkappa | $\Kappa(\kappa)\varkappa$ | \Lambda(\lambda)   | $\Lambda(\lambda)$   |
| \Nu(\nu)                      | $\Nu(\nu)$                      | \Xi(\xi)                | $\Xi(\xi)$                | \Omicron(\omicron) | $\Omicron(\omicron)$ |
| \Rho(\rho)\varrho             | $\Rho(\rho)\varrho$             | \Sigma(\sigma)\varsigma | $\Sigma(\sigma)\varsigma$ | \Tau(\tau)         | $\Tau(\tau)$         |
| \Phi(\phi)\varphi             | $\Phi(\phi)\varphi$             | \Chi(\chi)              | $\Chi(\chi)$              | \Psi(\psi)         | $\Psi(\psi)$         |
| \Delta(\delta)                | $\Delta(\delta)$                | \Theta(\theta)\vartheta | $\Theta(\theta)\vartheta$ | \Mu(\mu)           | $\Mu(\mu)$           |
| \Pi(\pi)\varpi                | $\Pi(\pi)\varpi$                | \Omega(\omega)          | $\Omega(\omega)$          | \upsilon           | $\upsilon$           |
| \ell                          | $\ell$                          | \eth                    | $\eth$                    | \hbar              | $\hbar$              |
| \hslash                       | $\hslash$                       | \mho                    | $\mho$                    | \partial           | $\partial$           |

### 12、特殊字符

#### ①说明
>可以在字符前使用`\large`或`\small`以显示更大或更小的字符。${\LARGE A}{\Large A}{\large A}A{\small A}$

#### ②关系运算符

| 输入      | 显示        | 输入              | 显示                | 输入           | 显示         |
| --------- | ----------- | ----------------- | ------------------- | -------------- | ------------ |
| \pm(\mp)  | $\pm(\mp)$ | \times            | $\times$            | \div           | $\div$       |
| \nmid     | $\nmid$     | \cdot             | $\cdot$             | \mid           | $\mid$       |
| \bigodot  | $\bigodot$  | \bigotimes        | $\bigotimes$        | \bigoplus      | $\bigoplus$  |
| \ge       | $\ge$       | \le               | $\le$               | \ll            | $\ll$        |
| \geqslant | $\geqslant$ | \leqslant         | $\leqslant$         | \neq           | $\neq$       |
| \approx   | $\approx$   | \xlongequal{文本} | $\xlongequal{文本}$ | \triangleq     | $\triangleq$ |
| \sim      | $\sim$      | \doteq            | $\doteq$            | \equiv         | $\equiv$     |
| \cong     | $\cong$     | \propto           | $\propto$           | \parallel(\\|) |
| \prec     | $\prec$     | \pmod{2}          | $\pmod{2}$          | \bmod          | $\bmod{2}$   |

#### ③集合运算符

| 输入      | 显示        | 输入        | 显示          | 输入       | 显示         |
| --------- | ----------- | ----------- | ------------- | ---------- | ------------ |
| \emptyset | $\emptyset$ | \varnothing | $\varnothing$ |            |              |
| \subset   | $\subset$   | \subseteq   | $\subseteq$   | \subsetneq | $\subsetneq$ |
| \supset   | $\supset$   | \supseteq   | $\supseteq$   | \supsetneq | $\supsetneq$ |
| \bigcap   | $\bigcap$   | \bigcup     | $\bigcup$     | \setminus  | $\setminus$  |
| \bigvee   | $\bigvee$   | \bigwedge   | $\bigwedge$   |            |              |
| \in       | $\in$       | \notin      | $\notin$      | \ni        | $\ni$        |

#### ④三角运算符

| 输入    | 显示      | 输入 | 显示   | 输入   | 显示     |
| ------- | --------- | ---- | ------ | ------ | -------- |
| \circ   | $\circ$   | \bot | $\bot$ | \angle | $\angle$ |
| \degree | $\degree$ |      |        |        |          |

#### ⑤微积分运算符

| 输入  | 显示    | 输入   | 显示     | 输入      | 显示     |
| ----- | ------- | ------ | -------- | --------- | -------- |
| \int  | $\int$  | \iint  | $\iint$  | \iiint    | $\iiint$ |
| \oint | $\oint$ | \oiint | $\oiint$ | \prime(‘) | $\prime$ |
| \lim  | $\lim$  | \infin | $\infin$ | \nabla    | $\nabla$ |
| \grad | $\grad$ |        |          |           |          |

#### ⑥逻辑运算符

| 输入     | 显示       | 输入       | 显示         | 输入   | 显示     |
| -------- | ---------- | ---------- | ------------ | ------ | -------- |
| \because | $\because$ | \therefore | $\therefore$ |        |          |
| \forall  | $\forall$  | \exist     | $\exist$     |        |          |
| \not>    | $\not>$    | \not<      | $\not<$      |        |          |
| \land    | $\land$    | \lor       | $\lor$       | \lnot  | $\lnot$  |
| \top     | $\top$     | \vdash     | $\vdash$     | \vDash | $\vDash$ |

#### ⑦带帽符号

| 输入           | 显示             | 输入            | 显示              |
| -------------- | ---------------- | --------------- | ----------------- |
| \hat{xy}       | $\hat{xy}$       | \widehat{xyz}   | $\widehat{xyz}$   |
| \tilde{xy}     | $\tilde{xy}$     | \widetilde{xyz} | $\widetilde{xyz}$ |
| \check{x}      | $\check{x}$      | \breve{y}       | $\breve{y}$       |
| \grave{x}      | $\grave{x}$      | \acute{y}       | $\acute{y}$       |
| \dot{x}        | $\dot{x}$        | \ddot{x}        | $\ddot{x}$        |
| \overparen{xy} | $\overparen{xy}$ |                 |                   |

#### ⑧选取符号

| 输入                           | 显示                             | 输入                            | 显示                              |
| ------------------------------ | -------------------------------- | ------------------------------- | --------------------------------- |
| \fbox{a+b+c+d}                 | $\fbox{a+b+c+d}$                 |                                 |                                   |
| \overbrace{xx\cdots x}^{10个x} | $\overbrace{xx\cdots x}^{10个x}$ | \underbrace{xx\cdots x}_{10个x} | $\underbrace{xx\cdots x}_{10个x}$ |


#### ⑨箭头符号

| 输入           | 显示             | 输入              | 显示                | 输入                | 显示                  |
| -------------- | ---------------- | ----------------- | ------------------- | ------------------- | --------------------- |
| \leftarrow     | $\leftarrow$     | \rightarrow       | $\rightarrow$       | \leftrightarrow     | $\leftrightarrow$     |
| \longleftarrow | $\longleftarrow$ | \longrightarrow   | $\longrightarrow$   | \longleftrightarrow | $\longleftrightarrow$ |
| \Leftarrow     | $\Leftarrow$     | \Rightarrow       | $\Rightarrow$       | \Leftrightarrow     | $\Leftrightarrow$     |
| \Longleftarrow | $\Longleftarrow$ | \Longrightarrow   | $\Longrightarrow$   | \Longleftrightarrow | $\Longleftrightarrow$ |
| \uparrow       | $\uparrow$       | \downarrow        | $\downarrow$        | \updownarrow        | $\updownarrow$        |
| \Uparrow       | $\Uparrow$       | \Downarrow        | $\Downarrow$        | \Updownarrow        | $\Updownarrow$        |
| \to            | $\to$            | \swarrow          | $\swarrow$          | \nearrow            | $\nearrow$            |
| \gets          | $\gets$          | \searrow          | $\searrow$          | \nwarrow            | $\nwarrow$            |
| \mapsto        | $\mapsto$        | \rightrightarrows | $\rightrightarrows$ |                     |                       |

#### ⑩空格

| 输入 | 效果   | 输入    | 效果   | 输入   | 效果       |
| ---- | ------ | ------- | ------ | ------ | ---------- |
| \\!  | $|\!|$ | 默认    | $||$   | \quad  | $|\quad|$  |
| \,   | $|\,|$ | \;(\\ ) | $|\;|$ | \qquad | $|\qquad|$ |

### 13、字体

> 代码:
>
> ```text
> ${\字体{需要转换的字符}}$
> ```

| 输入 | 说明 | 显示  | 输入  | 说明 | 显示 |
| ---- | ---- | ----- | ----- | ---- | ------ |
| \rm  | 罗马体 | ${\rm{Sample}}$ | \cal  | 花体 | ${\cal{Sample}}$ |
| \it  | 意大利体 | ${\it{Sample}}$ | \Bbb  | 黑板粗体 | ${\Bbb{Sample}}$ |
| \bf  | 粗体 | ${\bf{Sample}}$ | \mit  | 数学斜体 | ${\mit{Sample}}$ |
| \sf  | 等线体 | ${\sf{Sample}}$ | \scr  | 手写体 | ${\scr{Sample}}$ |
| \tt  | 打字机体 | ${\tt{Sample}}$ | \frak | 旧德式字体 | ${\frak{Sample}}$|

### 14、大括号和行标

>说明:  使用`\left`和`\right`来创建自动匹配高度的`()`、`[]`、`{}`、`.`。在每个公式末尾使用`\tag{行标}`来实现行标。

>代码:
>
>```text
>$$
>f\left(
>\left[
>\dfrac{1+\{x,y\}}{\left(\dfrac{x}{y}+\dfrac{y}{x}\right)(u+1)}+a
>\right]
>^{\dfrac{3}{2}}
>\right)
>\tag{行标}
>$$
>```

>效果:
>$$
>f\left(\left[\dfrac{1+\{x,y\}}{\left(\dfrac{x}{y}+\dfrac{y}{x}\right)(u+1)}+a\right]^{\dfrac{3}{2}}\right)\tag{行标}
>$$

>说明:如果你想将行内显示的分隔符也变大,也可以使用`\middle`命令

>代码:
>
>```text
>$$
>\left\langle q\middle\|\dfrac{\dfrac{x}{y}}{\dfrac{u}{v}}\middle|p\right\rangle
>$$
>```

>效果:
>$$
>\left\langle q\middle\|\dfrac{\dfrac{x}{y}}{\dfrac{u}{v}}\middle|p\right\rangle
>$$

### 15、其他命令

#### ①注释文字

>代码:
>```text
>$\text{文字}$
>```

>效果:
>$$
>f(n)=\begin{cases}n/2,&\text{if $n$ is even}\\3n+1,&\text{if $n$ is odd}\end{cases}
>$$

#### ③文字颜色

>* 适用新旧浏览器
> 代码:
> ```text
> $\color{颜色}{文字}$
> ```

| 输入    | 显示              | 输入   | 显示              | 输入   | 显示              |
| ------- | ----------------- | ------ | ----------------- | ------ | ----------------- |
| black   | $\color{black}{color}$ | grey   | $\color{grey}{color}$ | silver | $\color{silver}{color}$ |
| white   | $\color{white}{color}$ | maroon | $\color{maroon}{color}$ | red    | $\color{red}{color}$ |
| yellow  | $\color{yellow}{color}$ | lime   | $\color{lime}{color}$ | olive  | $\color{olive}{color}$ |
| green   | $\color{green}{color}$ | teal   | $\color{teal}{color}$ | auqa   | $\color{auqa}{color}$ |
| blue    | $\color{blue}{color}$ | navy   | $\color{navy}{color}$ | purple | $\color{purple}{color}$ |
| fuchsia | $\color{fuchsia}{color}$ |        |                   |        |                   |

>* 适用新版浏览器
>代码:
>```text
>$\color{#rgb}{文字}$    (注:其中r、g、b可以输入0~9和a~f来分别表示红色、绿色和蓝色的纯度)
>```

| 输入 | 输出              | 输入 | 输出              | 输入 | 输出              | 输入 | 输出              |
| ---- | ----------------- | ---- | ----------------- | ---- | ----------------- | ---- | ----------------- |
| #000 | $\color{#000}{color}$ | #005 | $\color{#005}{color}$ | #00A | $\color{#00A}{color}$ | #00F | $\color{#00F}{color}$ |
| #500 | $\color{#500}{color}$ | #505 | $\color{#505}{color}$ | #50A | $\color{#50A}{color}$ | #50F | $\color{#50F}{color}$ |
| #A00 | $\color{#A00}{color}$ | #A05 | $\color{#A05}{color}$ | #A0A | $\color{#A0A}{color}$ | #A0F | $\color{#A0F}{color}$ |
| #F00 | $\color{#F00}{color}$ | #F05 | $\color{#F05}{color}$ | #F0A | $\color{#F0A}{color}$ | #F0F | $\color{#F0F}{color}$ |
| #050 | $\color{#050}{color}$ | #055 | $\color{#055}{color}$ | #05A | $\color{#05A}{color}$ | #05F | $\color{#05F}{color}$ |
| #550 | $\color{#550}{color}$ | #555 | $\color{#555}{color}$ | #55A | $\color{#55A}{color}$ | #55F | $\color{#55F}{color}$ |
| #A50 | $\color{#A50}{color}$ | #A55 | $\color{#A55}{color}$ | #A5A | $\color{#A5A}{color}$ | #A5F | $\color{#A5F}{color}$ |
| #F50 | $\color{#F50}{color}$ | #F55 | $\color{#F55}{color}$ | #F5A | $\color{#F5A}{color}$ | #F5F | $\color{#F5F}{color}$ |
| #0A0 | $\color{#0A0}{color}$ | #0A5 | $\color{#0A5}{color}$ | #0AA | $\color{#0AA}{color}$ | #0AF | $\color{#0AF}{color}$ |
| #5A0 | $\color{#5A0}{color}$ | #5A5 | $\color{#5A5}{color}$ | #5AA | $\color{#5AA}{color}$ | #5AF | $\color{#5AF}{color}$ |
| #AA0 | $\color{#AA0}{color}$ | #AA5 | $\color{#AA5}{color}$ | #AAA | $\color{#AAA}{color}$ | #AAF | $\color{#AAF}{color}$ |
| #FA0 | $\color{#FA0}{color}$ | #FA5 | $\color{#FA5}{color}$ | #FAA | $\color{#FAA}{color}$ | #FAF | $\color{#FAF}{color}$ |
| #0F0 | $\color{#0F0}{color}$ | #0F5 | $\color{#0F5}{color}$ | #0FA | $\color{#0FA}{color}$ | #0FF | $\color{#0FF}{color}$ |
| #5F0 | $\color{#5F0}{color}$ | #5F5 | $\color{#5F5}{color}$ | #5FA | $\color{#5FA}{color}$ | #5FF | $\color{#5FF}{color}$ |
| #AF0 | $\color{#AF0}{color}$ | #AF5 | $\color{#AF5}{color}$ | #AFA | $\color{#AFA}{color}$ | #AFF | $\color{#AFF}{color}$ |
| #FF0 | $\color{#FF0}{color}$ | #FF5 | $\color{#FF5}{color}$ | #FFA | $\color{#FFA}{color}$ | #FFF | $\color{#FFF}{color}$ |

#### ③删除线

>说明:  使用`\require{cancle}`声明，再使用`\cancle{字符}`、`\bcancle{字符}`、`\xcancle{字符}`、`\cancleto{字符}{字符}`来实现各种**片段删除线**效果。

>代码:
>```text
>$$
>\require{cancel}\begin{array}{r1}
>\verb|y+\cancel{x}|&y+\cancel{x}\\
>\verb|y+\cancel{y+x}|&y+\cancel{y+x}\\
>\verb|y+\bcancel{x}|&y+\bcancel{x}\\
>\verb|y+\xcancel{x}|&y+\xcancel{x}\\
>\verb|y+\cancelto{0}{x}|&y+\cancelto{0}{x}\\
>\verb+\frac{1\cancel9}{\cancel95}=\frac15+&\frac{1\cancel9}{\cancel95}=\frac15\\
>\end{array}
>$$
>```

>效果:
>$$
>\require{cancel}\begin{array}{r1}
>\verb|y+\cancel{x}|&y+\cancel{x}\\
>\verb|y+\cancel{y+x}|&y+\cancel{y+x}\\
>\verb|y+\bcancel{x}|&y+\bcancel{x}\\
>\verb|y+\xcancel{x}|&y+\xcancel{x}\\
>\verb|y+\cancelto{0}{x}|&y+\cancelto{0}{x}\\
>\verb+\frac{1\cancel9}{\cancel95}=\frac15+&\frac{1\cancel9}{\cancel95}=\frac15\\
>\end{array}
>$$

>说明:  使用`\require{enclose}`来允许**整段删除线**的显示，再使用`\enclose{删除线效果}{字符}`来使用各种整段删除线效果。其中，删除线效果有`horizontalstrike`、`verticalstrike`、`updiagonalstrike`和`downdiagonalstrike`,可以叠加使用。

>代码:
>```text
>$$
>\require{enclose}\begin{array}{r1}
>\verb|\enclose{horizontalstrike}{x+y}|&\enclose{horizontalstrike}{x+y}\\
>\verb|\enclose{verticalstrike}{\frac xy}|&\enclose{verticalstrike}{\frac xy}\\
>\verb|\enclose{updiagonalstrike}{x+y}|&\enclose{updiagonalstrike}{x+y}\\
>\verb|\enclose{downdiagonalstrike}{x+y}|&\enclose{downdiagonalstrike}{x+y}\\
>\verb|\enclose{horizontalstrike,updiagonalstrike}{x+y}|&\enclose{horizontalstrike,updiagonalstrike}{x+y}\\
>\end{array}
>$$
>```

>效果:
>$$
>\require{enclose}\begin{array}{r1}
>\verb|\enclose{horizontalstrike}{x+y}|&\enclose{horizontalstrike}{x+y}\\
>\verb|\enclose{verticalstrike}{\frac xy}|&\enclose{verticalstrike}{\frac xy}\\
>\verb|\enclose{updiagonalstrike}{x+y}|&\enclose{updiagonalstrike}{x+y}\\
>\verb|\enclose{downdiagonalstrike}{x+y}|&\enclose{downdiagonalstrike}{x+y}\\
>\verb|\enclose{horizontalstrike,updiagonalstrike}{x+y}|&\enclose{horizontalstrike,updiagonalstrike}{x+y}\\
>\end{array}
>$$

### 16、矩阵

#### ①无框矩阵

>代码:
>```text
>$$
>\begin{matrix}
>1&x&x^2\\
>1&y&y^2\\
>1&z&z^2\\
>\end{matrix}
>$$
>```

>效果:
>$$
>\begin{matrix}
>1&x&x^2\\
>1&y&y^2\\
>1&z&z^2\\
>\end{matrix}
>$$

#### ②边框矩阵

>说明:  在开头将`matrix`替换为`pmatrix`、`bmatrix`、`Bmatrix`、`vmatrix`、`Vmatrix`。

| matrix                               | pmatrix                                | bmatrix                                | Bmatrix                                | vmatrix                                | Vmatrix                                |
| ------------------------------------ | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| $\begin{matrix}1&2\\3&4\end{matrix}$ | $\begin{pmatrix}1&2\\3&4\end{pmatrix}$ | $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ | $\begin{Bmatrix}1&2\\3&4\end{Bmatrix}$ | $\begin{vmatrix}1&2\\3&4\end{vmatrix}$ | $\begin{Vmatrix}1&2\\3&4\end{Vmatrix}$ |

#### ③带分割线的矩阵

>说明:  可以使用`cc|c`来在一个三列矩阵中插入分割线。

>代码:
>```text
>$$
>\left[
>\begin{array}{cc|c}
>1&2&3\\
>4&5&6
>\end{array}
>\right]
>$$
>```

>效果:
>$$
>\left[
>\begin{array}{cc|c}
>1&2&3\\
>4&5&6
>\end{array}
>\right]
>$$

#### ④行中矩阵

>代码:
>```text
>$\bigl(\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\bigr)$
>```

>效果:
>$\bigl(\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\bigr)$

### 17、方程式序列

>说明:  可以使用`\begin{align}...\end{align}`来创建一列整齐且默认右对齐的方程式序列。请注意`{align}`是**自动编号**的，使用`{align*}`来声明停止自动编号，也可以使用`\notag`来取消特定行的自动编号。在需要的时候，你可以使用`\begin{equation}...\end{equation}`来强制表达式自动编号。

>代码:
>$$
>\begin{align}
>f(x)&=1+1\\
>&=2
>\end{align}
>$$
>
>$$
>\begin{equation}
>\left[
>\begin{array}{cc|c}
>1&2&3\\
>4&5&6
>\end{array}
>\right]
>\end{equation}
>$$
>
>
>
>```text
>$$
>\begin{align}
>\sqrt{37}=\sqrt{\dfrac{73^2-1}{12^2}}\\
>&=\sqrt{\dfrac{73^2}{12^2}\cdot\dfrac{73^2-1}{73^2}}\\
>&=\sqrt{\dfrac{73^2}{12^2}}\sqrt{\dfrac{73^2-1}{73^2}}\notag\\
>&=\dfrac{73}{12}\sqrt{1-\dfrac{1}{73^2}}\\
>\approx\dfrac{73}{12}\left(1-\dfrac{1}{2\cdot73^2}\right)\label{A}
>\end{align}
>$$
>***
>
>$$
>\begin{align*}
>v+m&=0&\text{Given}\tag1\\
>-w&=-w+0&\text{additive identity}\tag2\\
>-w+0&=-w+(v+w)&\text{equations $(1)$ and $(2)$}
>\end{align*}
>$$
>```

>效果:
>$$
>\begin{align}
>\sqrt{37}&=\sqrt{\dfrac{73^2-1}{12^2}}\\
>&=\sqrt{\dfrac{73^2}{12^2}\cdot\dfrac{73^2-1}{73^2}}\\
>&=\sqrt{\dfrac{73^2}{12^2}}\sqrt{\dfrac{73^2-1}{73^2}}\notag\\
>&=\dfrac{73}{12}\sqrt{1-\dfrac{1}{73^2}}\\
>&\approx\dfrac{73}{12}\left(1-\dfrac{1}{2\cdot73^2}\right)\label{A}
>\end{align}
>$$
>***
>
>$$
>\begin{align*}
>v+m&=0&\text{Given}\tag1\\
>-w&=-w+0&\text{additive identity}\tag2\\
>-w+0&=-w+(v+w)&\text{equations $(1)$ and $(2)$}
>\end{align*}
>$$
>
>你可以使用`\label{标签}`来创建一个标签，就如上面的方程式序列中展示的那样，之后使用`\eqref{标签}`引用你想引用的公式，效果为：$\eqref{A}$。如果不想要括号，可以输入`\ref{标签}`，效果为：公式 $\ref{A}$。
>
>公式1和2的不同列之间存在间隔，如果你不想要，可以通过将`align`替换为`alignat{1}`来去除列间隔。

### 18、条件表达式

>说明:  使用`\begin{cases}`来创造一组默认左对齐的条件表达式,在每一行插入`&`来指定需要对齐的内容,并在每一行结尾处使用`\\`,以`\end{cases}`结尾。

>代码:
>```text
>$$
>f(n)=
>\begin{cases}
>n/2,&\text{if $n$ is even}\\
>3n+1,&\text{if $n$ is odd}
>\end{cases}
>$$
>```

>效果:
>$$
>f(n)=
>\begin{cases}
>n/2,&\text{if $n$ is even}\\
>3n+1,&\text{if $n$ is odd}
>\end{cases}
>$$

### 19、配置行高

>说明:  可以使用`\\[2ex]`语句替代该行末尾的`\\`来让编译器适配 , 其中`[ex]`指一个"X-Height" , 即x字母高度 , 也可以使用`[3ex]`或`[4ex]`等。

>代码:
>```text
>$$
>f(n)=
>\begin{cases}
>\dfrac n2,&\text{if $n$ is even}\\[2ex]
>3n+1,&\text{if $n$ is odd}
>\end{cases}\tag{适配[2ex]}
>$$
>***
>
>$$
>f(n)=
>\begin{cases}
>\dfrac n2,&\text{if $n$ is even}\\
>3n+1,&\text{if $n$ is odd}
>\end{cases}\tag{不适配[2ex]}
>$$
>```

>效果:
>$$
>f(n)=
>\begin{cases}
>\dfrac n2,&\text{if $n$ is even}\\[2ex]
>3n+1,&\text{if $n$ is odd}
>\end{cases}\tag{适配[2ex]}
>$$
>***
>
>$$
>f(n)=
>\begin{cases}
>\dfrac n2,&\text{if $n$ is even}\\
>3n+1,&\text{if $n$ is odd}
>\end{cases}\tag{不适配[2ex]}
>$$

### 20、数组与表格

>说明:  数组与表格均以`\begin{array}`开头,并在其后定义列数及每一列的文本对齐方式,`c` `l` `r`分别代表居中、左对齐及右对齐。若要插入垂直分割线，在定义中插入`|`，若要插入水平分割线，在定义中加入`\hline`。

>代码:
>
>```text
>$$
>\begin{array}{c|lcr}
>n&\text{左对齐}&\text{居中对齐}&\text{右对齐}\\
>\hline
>1&0.24&1&125\\
>2&-1&189&-8\\
>3&-20&2000&1+10i
>\end{array}
>$$
>```

>效果:
>$$
>\begin{array}{c|lcr}
>n&\text{左对齐}&\text{居中对齐}&\text{右对齐}\\
>\hline
>1&0.24&1&125\\
>2&-1&189&-8\\
>3&-20&2000&1+10i
>\end{array}
>$$

### 21、嵌套表格或数组

>代码:
>```text
>$$
>% outer vertical array of arrays 外层垂直表格
>\begin{array}{c}
>% inner horizontal array of arrays 内层水平表格
>\begin{array}{cc}
>% inner array of minimum values 内层"最小值"数组
>\begin{array}{c|cccc}
>\text{min}&0&1&2&3\\
>\hline
>0&0&0&0&0\\
>1&0&1&1&1\\
>2&0&1&2&2\\
>3&0&1&2&3\\
>\end{array}
>&
>% inner array of maximum values 内层"最大值"数组
>\begin{array}{c|cccc}
>\text{max}&0&1&2&3\\
>\hline
>0&0&1&2&3\\
>1&1&1&2&3\\
>2&2&2&2&3\\
>3&3&3&3&3
>\end{array}
>\end{array}
>% 内层第一行表格组结束
>\\
>% inner array of delta values 内层第二行Delta值数组
>\begin{array}{c|cccc}
>\Delta&0&1&2&3\\
>\hline
>0&0&1&2&3\\
>1&1&0&1&2\\
>2&2&1&0&1\\
>3&3&2&1&0
>\end{array}
>% 内层第二行表格组结束
>\end{array}
>$$
>```

>效果:
>$$
>% outer vertical array of arrays 外层垂直表格
>\begin{array}{c}
>% inner horizontal array of arrays 内层水平表格
>\begin{array}{cc}
>% inner array of minimum values 内层"最小值"数组
>\begin{array}{c|cccc}
>\text{min}&0&1&2&3\\
>\hline
>0&0&0&0&0\\
>1&0&1&1&1\\
>2&0&1&2&2\\
>3&0&1&2&3\\
>\end{array}
>&
>% inner array of maximum values 内层"最大值"数组
>\begin{array}{c|cccc}
>\text{max}&0&1&2&3\\
>\hline
>0&0&1&2&3\\
>1&1&1&2&3\\
>2&2&2&2&3\\
>3&3&3&3&3
>\end{array}
>\end{array}
>% 内层第一行表格组结束
>\\
>% inner array of delta values 内层第二行Delta值数组
>\begin{array}{c|cccc}
>\Delta&0&1&2&3\\
>\hline
>0&0&1&2&3\\
>1&1&0&1&2\\
>2&2&1&0&1\\
>3&3&2&1&0
>\end{array}
>% 内层第二行表格组结束
>\end{array}
>$$

### 22、方程组

>说明:  使用`\begin{array}...\end{array}`和`\left\{...\right.`来创建一个方程组,或者你也可以使用条件表达式组`\begin{cases}...\end{cases}`来实现相同效果。

>代码:
>```text
>$$
>\left\{
>\begin{array}{l}
>a_1x+b_1y+c_1z=d_1\\
>a_2x+b_2y+c_2z=d_2\\
>a_3x+b_3y+c_1z=d_3
>\end{array}
>\right.
>\quad\text{或者}\quad
>\begin{cases}
>a_1x+b_1y+c_1z=d_1\\
>a_2x+b_2y+c_2z=d_2\\
>a_3x+b_3y+c_1z=d_3
>\end{cases}
>$$
>```

>效果:
>$$
>\left\{
>\begin{array}{l}
>a_1x+b_1y+c_1z=d_1\\
>a_2x+b_2y+c_2z=d_2\\
>a_3x+b_3y+c_1z=d_3
>\end{array}
>\right.
>\quad\text{或者}\quad
>\begin{cases}
>a_1x+b_1y+c_1z=d_1\\
>a_2x+b_2y+c_2z=d_2\\
>a_3x+b_3y+c_1z=d_3
>\end{cases}
>$$

### 23、连分式

>说明:  就像`\frac`一样,使用`\cfrac`或`\dfrac`来创建一个连分式,不要使用普通的`\frac`或`\over`来创建,否则看起来会**很恶心**。

>代码:
>```text
>$$
>x=a_0+\cfrac{1^2}{a_1+\cfrac{2^2}{a_2+\cfrac{3^2}{a_3+\cfrac{4^2}{a_4+\cdots}}}}
>$$
>```

>效果:
>$$
>x=a_0+\cfrac{1^2}{a_1+\cfrac{2^2}{a_2+\cfrac{3^2}{a_3+\cfrac{4^2}{a_4+\cdots}}}}
>$$

>反例:
>```text
>x=a_0+\frac{1^2}{a_1+\frac{2^2}{a_2+\frac{3^2}{a_3+\frac{4^2}{a_4+\cdots}}}}
>```

>效果:
>$$
>x=a_0+\frac{1^2}{a_1+\frac{2^2}{a_2+\frac{3^2}{a_3+\frac{4^2}{a_4+\cdots}}}}
>$$

>补充:  当然,你可以使用`\frac`来表达连分数的**紧缩记法**。

>代码:
>```text
>$$
>x=a_0+\frac{1^2}{a_1+}\frac{2^2}{a_2+}\frac{3^2}{a_3+}\frac{4^2}{a_4+}\cdots
>$$
>```

>效果:
>$$
>x=a_0+\frac{1^2}{a_1+}\frac{2^2}{a_2+}\frac{3^2}{a_3+}\frac{4^2}{a_4+}\cdots
>$$

### 24、交换图表

>说明:  使用一行`$\require{AMScd}$`语句来允许交换图表的显示,并通过在开头使用`\begin{CD}`,结尾使用`\end{CD}`来创建。

>代码:
>```text
>$$
>\require{AMScd}
>\begin{CD}
>A@>a>>B\\
>@VbVV\# @VcVV\\
>C @>>d> D
>\end{CD}
>$$
>```

>效果:
>$$
>\require{AMScd}
>\begin{CD}
>A@>a>>B\\
>@V b V V\# @VV c V\\
>C @>>d> D
>\end{CD}
>$$

>补充:  其中,`@>>>`代表右箭头、`@<<<`代表左箭头、`@VVV`代表下箭头、`@AAA`代表上箭头、`@=`代表水平双实线、`@|`代表竖直双实线、`@.`代表没有箭头。在`@>>>`的`>>>`之间任意插入文字即代表该箭头的注释文字。

>代码:
>```text
>$$
>\begin{CD}
>A@>>>B@>{\text{very long label}}>>C\\
>@.@AAA@|\\
>D@=E@<<<F
>\end{CD}
>$$
>```

>效果:
>$$
>\begin{CD}
>A@>>>B@>{\text{very long label}}>>C\\
>@.@AAA@|\\
>D@=E@<<<F
>\end{CD}
>$$

### 25、其他

>* 搜索$\LaTeX$

## 十四、支持的HTML元素

### 1、文本居中
>代码
>```text
><center>内容</center>
>```

>效果
>
><center>内容</center>

### 2、快捷键显示

>代码:
>```text
><kbd>内容</kbd>
>```

>效果:
><kbd>内容</kbd>

### 3、加粗

>代码:
>
>```text
><b>加粗</b>
>```

>效果:
>
><b>加粗</b>

### 4、倾斜
>代码:
>```text
><i>倾斜</i>
>```

>效果:
><i>倾斜</i>

### 5、上下标

>代码:
>```text
>开始<sup>123hi你好</sup>
>开始<sub>321hi你好</sub>
>```

>效果:
>开始<sup>123hi你好</sup>
>开始<sub>321hi你好</sub>

### 6、填充的黑色箭头

> 代码：
>
> ```text
> &#x27A4;
> ```

> 效果：
> &#x27A4;
