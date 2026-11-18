# VPS 代理：架构说明与部署指南

> 版本日期 2026-11-18。第一部分是架构；第二部分是部署指南，面向 agent 执行，标注 `[HUMAN]` 的步骤需由操作人员完成。

# 第一部分：架构（给人看的，简要介绍）

这套系统运行在一台境外云服务器（VPS）上，提供三项功能：

- **代理**：电脑上的客户端（Clash Verge Rev）把网络请求发给 VPS，由 VPS 转发出去。目标网站看到的访问来源是一个美国的固定 IP。
- **Claude 网页版镜像**：用浏览器打开自己的域名就能使用 claude.ai，本机不需要任何代理设置。
- **管理**：通过网页面板管理用户和流量配额，并生成客户端导入用的订阅链接。

![VPS 代理架构](../Preview/vpn-architecture.png)

图中蓝色实线是代理流量，紫色实线是镜像流量，灰色虚线是管理流量。

**统一入口。** 代理和镜像的请求都从 VPS 的 443 端口进入。443 是 HTTPS 网站的标准端口，所以这些连接从外部看和普通的网页访问一样。

sing-box 监听 443。每个连接开头会声明要访问的域名，这个字段叫 SNI。sing-box 只读取这个字段来决定把连接交给谁，不解密内容：

- **SNI 是 `swdist.apple.com`**：交给 Xray。Xray 先确认对方是本系统的用户，再转发请求。如果不是用户，比如外部的探测，Xray 会把连接原样转给真正的 swdist.apple.com，对方看到的就是苹果的网站。
- **SNI 是自有域名**：交给 Caddy。Caddy 持有这些域名的 HTTPS 证书，负责解密，再按域名把请求转给 Claude 镜像（reclaude）或管理面板（Marzban）。

**出口。** VPS 不直接访问目标网站，而是经 decodo 提供的美国运营商 IP 出去。云服务器的 IP 属于数据中心，容易被网站限制；运营商 IP 登记在运营商名下，和家庭宽带属于同一类，受到的限制较少。代理可以固定用一个 IP，也可以在几个 IP 之间随机分配。Claude 镜像始终固定用同一个 IP，这样同一账号的访问来源不会频繁变化。

**管理。** Marzban 保存用户、配额和订阅信息，并根据这些信息生成 Xray 的配置。管理员在 `https://<域名>:8443` 打开面板，客户端也从这个地址更新订阅。Xray、Marzban、reclaude 都只在 VPS 内部监听，外部无法直接访问。

---

# 第二部分：部署指南（直接贴给agent看即可）

## 执行规则

1. **按顺序执行**。每一步末尾都有「验证」。未通过就停止，把输出提交给操作人员。不得跳过步骤，也不得自行更换方案。
2. **`[HUMAN]` 步骤由操作人员完成**。agent 在此停止，说明需要做什么，等操作人员确认后再继续。
3. **以 root 执行每个代码块**，比如保存成文件后用 `sudo bash step.sh` 运行。每个代码块都能单独运行，开头会自己加载 `/etc/proxy-deploy/deploy.env`。
4. **不要在输出里打印秘密**，包括密码、私钥、UUID。这些只写进 `/etc/proxy-deploy/` 下权限为 600 的文件。例外：第 17 步的订阅链接要提交给操作人员，用于导入客户端。
5. **覆盖已有文件前先备份**，存成 `<文件>.bak-<时间戳>`。
6. **端口被占用或服务已存在时停下汇报**，不要杀进程，也不要卸载东西。
7. **目标环境**：一台全新的 Ubuntu 22.04 或 24.04（x86_64），当前用户有 sudo 权限。

## 资源准备 `[HUMAN]`

本节由操作人员在交给 agent 之前完成。完成后得到的值填入第 0 步的输入参数。

### A. 腾讯云 VPS

1. 注册腾讯云账号，完成实名认证。VPS 在境外地域，不需要 ICP 备案。
2. 购买云服务器 CVM，参数如下：

| 参数     | 取值                                                                                            |
| -------- | ----------------------------------------------------------------------------------------------- |
| 地域     | 新加坡；也可以选中国香港、日本东京等离客户端较近的境外地域                                      |
| 实例规格 | 2 vCPU / 2 GB 及以上（Claude 镜像约占 200 MB 内存）；1 GB 内存的实例需要配置 swap |
| 镜像     | 公共镜像 Ubuntu Server 22.04 LTS 64 位，不要选应用镜像                                          |
| 公网 IP  | 分配独立公网 IPv4；带宽计费方式和带宽上限按用量选择                                             |
| 登录方式 | 绑定 SSH 密钥对；Ubuntu 镜像的默认用户是`ubuntu`                                              |

   如果想用更便宜的方案，也可以选「轻量应用服务器」的新加坡地域，它自带流量包。轻量应用服务器里对应的功能叫「防火墙」，配置方式和下面的安全组相同。
3. 配置安全组的入站规则。这一步要覆盖默认规则，只保留下表这几条：

| 端口 | 协议 | 来源                  | 用途          |
| ---- | ---- | --------------------- | ------------- |
| 22   | TCP  | 运维来源 IP           | SSH           |
| 80   | TCP  | 0.0.0.0/0             | ACME 证书签发 |
| 443  | TCP  | 0.0.0.0/0             | 代理和面板    |
| 8443 | TCP  | 0.0.0.0/0，或限定来源 | 面板和订阅    |

4. 记录实例的公网 IP，对应第 0 步的 `VPS_IP`。

### B. 域名与 DNS

1. 在任意注册商购买域名，比如腾讯云 DNSPod、Cloudflare、Namesilo。
2. 添加一条 A 记录：`DOMAIN` → `VPS_IP`，TTL 600。
3. 为 Claude 镜像再添加 `claude.<DOMAIN>`、`content.<DOMAIN>`、`claude-mcp.<DOMAIN>` 三条 A 记录，同样指向 `VPS_IP`；也可以用一条 `*.<DOMAIN>` 通配记录代替。
4. 如果 DNS 托管在 Cloudflare，以上记录的代理状态都必须设为「DNS only」（灰色云朵）。经 Cloudflare 代理会导致 SNI 分流和 REALITY 失效。
5. 部署 claude-relay-service 时（可选步骤 A），再加一条 `relay.<DOMAIN>` 的 A 记录。

### C. decodo ISP 代理（`EXIT_MODE=direct` 时跳过）

1. 在 [decodo.com](https://decodo.com) 注册账号，购买 Dedicated ISP 代理：地区选 United States，数量按需（在用环境是 3 个 IP）。
2. 进入控制台 Static Residential → Dedicated ISP → Proxy setup：
   - 认证方式：选择用户名 + 密码。第一个代理用户会自动创建，在 Authentication 页可以查看和新增用户；
   - Session type：选 Static；
   - 协议：选 SOCKS5（HTTP 也可以）。
3. 端口规则：
   - 入口地址都是 `isp.decodo.com`；
   - 端口 10000 每次请求都会更换 IP，**不要用**；
   - 端口 10001–63000 是固定 IP 会话。
   - 端口和 IP 的对应关系以 Proxy setup 页生成的列表为准，购买的 IP 会循环出现在列表里。在用环境中，10001、10002、10003 分别对应 3 个不同的 IP。
4. 选定要用的端口，逐个验证它们的出口 IP：

```bash
for p in 10001 10002 10003; do
  echo "$p -> $(curl -s -U "$DECODO_USER:$DECODO_PASS" -x "socks5h://isp.decodo.com:$p" https://ip.decodo.com/json)"
done
```

5. 选好的端口填入 `DECODO_PORTS`，其中用作固定出口的那个填入 `FIXED_PORT`。

参考：[Dedicated ISP 快速入门](https://help.decodo.com/docs/dedicated-isp-proxy-quick-start)、[认证方式](https://help.decodo.com/docs/dedicated-isp-proxy-authentication-methods)。

### D. 客户端

客户端从 [Clash Verge Rev 的 GitHub Releases](https://github.com/clash-verge-rev/clash-verge-rev/releases) 下载安装。部署完成后，按第 18 步导入订阅并配置终端。

### E. 交给 agent

1. 提供 agent 登录 VPS 的方式：SSH 密钥，用户 `ubuntu`，需要有 sudo 权限。
2. 提供第 0 步的输入参数：域名、IP、出口模式、decodo 的账号密码和端口、第一个用户的用户名。

## 第 0 步：输入参数

| 变量                              | 含义                                                                            | 示例                  | 谁提供                                                  |
| --------------------------------- | ------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------- |
| `DOMAIN`                        | 域名，已解析到 VPS                                                              | `proxy.example.com` | 操作人员                                                |
| `VPS_IP`                        | VPS 公网 IPv4                                                                   | `203.0.113.10`      | 操作人员（agent 可以用`curl -4 -s ifconfig.me` 核对） |
| `REALITY_DEST`                  | REALITY 伪装的网站                                                              | `swdist.apple.com`  | 默认值                                                  |
| `EXIT_MODE`                     | 出口方式：`fixed` 固定端口 / `random` 随机轮换 / `direct` 由 VPS 直接出站。使用 Claude Code 时选 `fixed` | `fixed`             | 操作人员                                                |
| `DECODO_USER` / `DECODO_PASS` | decodo 的账号和密码                                                             | —                    | 操作人员（`direct` 模式不需要）                       |
| `DECODO_PORTS`                  | 要用的 decodo 端口，逗号分隔                                                    | `10001,10002,10003` | 操作人员                                                |
| `FIXED_PORT`                    | 固定出口端口，须在 `DECODO_PORTS` 中：`fixed` 模式的代理出口，以及 Claude 镜像的出口 | `10002`             | 操作人员                                                |
| `FIRST_USER`                    | 第一个代理用户的用户名                                                          | `alice`             | 操作人员                                                |

agent 把操作人员提供的值写入下面的文件：

```bash
install -d -m 700 /etc/proxy-deploy
cat > /etc/proxy-deploy/deploy.env <<'EOF'
DOMAIN=proxy.example.com
VPS_IP=203.0.113.10
REALITY_DEST=swdist.apple.com
EXIT_MODE=fixed
DECODO_USER=
DECODO_PASS=
DECODO_PORTS=10001,10002,10003
FIXED_PORT=10002
FIRST_USER=alice
EOF
chmod 600 /etc/proxy-deploy/deploy.env
```

这个文件会被 bash 直接加载（`source`）。如果某个值里有空格、`$` 之类的特殊字符，要用单引号括起来，比如 `DECODO_PASS='p@ss$word'`。

## 第 1 步：前置条件

`[HUMAN]` 确认「资源准备」的 A、B、C 三项都已完成（`EXIT_MODE=direct` 时可以跳过 C）。agent 无法从 VPS 内部检查安全组的配置；如果安全组没配好，问题会在第 4 步（证书签发）或客户端连接时暴露出来。

**agent 检查：**

```bash
set -euo pipefail; . /etc/proxy-deploy/deploy.env
. /etc/os-release; echo "$PRETTY_NAME $(uname -m)"
[ "$(getent ahostsv4 "$DOMAIN" | awk 'NR==1{print $1}')" = "$VPS_IP" ] && echo DNS_OK || echo DNS_MISMATCH
S=SUB_DNS_OK; for h in claude content claude-mcp; do
  [ "$(getent ahostsv4 "$h.$DOMAIN" | awk 'NR==1{print $1}')" = "$VPS_IP" ] || S="SUB_DNS_MISMATCH($h)"
done; echo $S
ss -tln | grep -E ':(80|443|8443|8000|8444|4000|4001|4010|4020) ' && echo PORTS_BUSY || echo PORTS_FREE
```

**验证**：输出里要有 `DNS_OK`、`SUB_DNS_OK` 和 `PORTS_FREE`。

## 第 2 步：系统初始化

```bash
set -euo pipefail
cat > /etc/sysctl.d/90-proxy.conf <<'EOF'
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
EOF
sysctl --system >/dev/null
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y curl jq openssl python3 tcpdump ca-certificates gnupg
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
VER=1.13.12; cd /tmp
curl -fsSLO https://github.com/SagerNet/sing-box/releases/download/v$VER/sing-box-$VER-linux-amd64.tar.gz
tar xzf sing-box-$VER-linux-amd64.tar.gz
install -m 755 sing-box-$VER-linux-amd64/sing-box /usr/local/bin/
```

**验证**：

- `sysctl -n net.ipv4.tcp_congestion_control` 输出 `bbr`；
- `docker version --format '{{.Server.Version}}'` 能输出版本号；
- `sing-box version | head -1` 能输出版本号。

## 第 3 步：生成密钥

本步骤可以重复执行，已经生成的密钥不会被覆盖。

```bash
set -euo pipefail; F=/etc/proxy-deploy/deploy.env
if ! grep -q '^REALITY_PRIVATE_KEY=' "$F"; then
  KP=$(sing-box generate reality-keypair)
  echo "REALITY_PRIVATE_KEY=$(echo "$KP" | awk '/PrivateKey/{print $2}')" >> "$F"
  echo "REALITY_PUBLIC_KEY=$(echo "$KP" | awk '/PublicKey/{print $2}')" >> "$F"
  echo "REALITY_SHORT_ID=$(openssl rand -hex 8)" >> "$F"
  echo "MARZBAN_ADMIN_USER=admin" >> "$F"
  echo "MARZBAN_ADMIN_PASS=$(openssl rand -hex 16)" >> "$F"
fi
grep -cE '^(REALITY_PRIVATE_KEY|REALITY_PUBLIC_KEY|REALITY_SHORT_ID|MARZBAN_ADMIN_PASS)=.+' "$F"
```

**验证**：输出 `4`。

## 第 4 步：Caddy（证书 + 面板入口）

```bash
set -euo pipefail; . /etc/proxy-deploy/deploy.env
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
apt-get update -y && apt-get install -y caddy
[ -f /etc/caddy/Caddyfile ] && cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak-$(date +%s)
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN:8443 {
    reverse_proxy 127.0.0.1:8000
}
EOF
systemctl reload caddy || systemctl restart caddy
for i in $(seq 24); do
  journalctl -u caddy --since -10min | grep -q 'certificate obtained successfully' && { echo CERT_OK; break; }
  sleep 5
done
```

**验证**：输出 `CERT_OK`。没有出现时，常见原因是安全组没有放行 TCP 80，或者 DNS 记录有误。此时停止，交由操作人员处理（`[HUMAN]`）。

## 第 5 步：部署 Marzban

部署的是 [Gozargah/Marzban](https://github.com/Gozargah/Marzban)，用官方安装脚本。该脚本最后会执行 `logs -f` 跟踪日志，不会自行退出。所以在后台运行它，等容器启动后，再终止日志跟踪进程。

```bash
set -euo pipefail
[ -d /opt/marzban ] && { echo "Marzban 已存在，停止"; exit 1; }
curl -fsSL https://github.com/Gozargah/Marzban-scripts/raw/master/marzban.sh -o /tmp/marzban.sh
nohup bash /tmp/marzban.sh install </dev/null >/tmp/marzban-install.log 2>&1 &
for i in $(seq 120); do
  docker ps --format '{{.Names}}' | grep -qx marzban-marzban-1 && break
  sleep 5
done
sleep 5
pkill -f 'opt/marzban/docker-compose.yml.*logs -f' || true
pkill -f '/tmp/marzban.sh install' || true
docker ps --format '{{.Names}} {{.Status}}' | grep marzban-marzban-1
command -v marzban
```

**验证**：

- 能看到 `marzban-marzban-1 Up ...`；
- `command -v marzban` 输出 `/usr/local/bin/marzban`。

如果失败，查看 `/tmp/marzban-install.log`。

## 第 6 步：配置 Marzban 和 Xray

这一步做三件事：

- 写 `.env`：面板只监听本机，管理员账号用环境变量设置（Marzban 支持 `SUDO_USERNAME` 和 `SUDO_PASSWORD`）；
- 生成 Xray 配置：REALITY 入站，出口按 `EXIT_MODE` 设置；
- 重启 Marzban。

```bash
set -euo pipefail; set -a; . /etc/proxy-deploy/deploy.env; set +a
E=/opt/marzban/.env; cp $E $E.bak-$(date +%s)
setenv(){ sed -i -E "/^[#[:space:]]*$1[[:space:]]*=/d" "$E"; echo "$1 = $2" >> "$E"; }
setenv UVICORN_HOST '"127.0.0.1"'
setenv UVICORN_PORT 8000
setenv DASHBOARD_PATH '"/dashboard/"'
setenv XRAY_JSON '"/var/lib/marzban/xray_config.json"'
setenv XRAY_SUBSCRIPTION_URL_PREFIX "\"https://$DOMAIN:8443\""
setenv SUDO_USERNAME "\"$MARZBAN_ADMIN_USER\""
setenv SUDO_PASSWORD "\"$MARZBAN_ADMIN_PASS\""

X=/var/lib/marzban/xray_config.json; [ -f $X ] && cp $X $X.bak-$(date +%s)
python3 - > /tmp/xray_config.json <<'PY'
import json, os
e = os.environ
mode = e.get("EXIT_MODE", "direct")
ports = [p.strip() for p in e.get("DECODO_PORTS", "").split(",") if p.strip()]
inbound = {
  "tag": "VLESS_REALITY_INBOUND", "listen": "127.0.0.1", "port": 8444, "protocol": "vless",
  "settings": {"clients": [], "decryption": "none"},
  "streamSettings": {"network": "tcp", "security": "reality", "realitySettings": {
    "show": False, "dest": e["REALITY_DEST"] + ":443", "xver": 0,
    "serverNames": [e["REALITY_DEST"]], "privateKey": e["REALITY_PRIVATE_KEY"],
    "shortIds": [e["REALITY_SHORT_ID"]]}},
  "sniffing": {"enabled": True, "destOverride": ["http", "tls", "quic"]}}
outbounds = [{"protocol": "freedom", "tag": "DIRECT"}, {"protocol": "blackhole", "tag": "BLOCK"}]
if mode in ("fixed", "random"):
  for p in ports:
    outbounds.append({"protocol": "socks", "tag": f"US_ISP_{p}", "settings": {"servers": [{
      "address": "isp.decodo.com", "port": int(p),
      "users": [{"user": e["DECODO_USER"], "pass": e["DECODO_PASS"]}]}]}})
rules = [{"type": "field", "ip": ["geoip:private"], "outboundTag": "BLOCK"}]
routing = {"domainStrategy": "IPIfNonMatch", "rules": rules}
if mode == "fixed":
  rules.append({"type": "field", "network": "tcp,udp", "outboundTag": f"US_ISP_{e['FIXED_PORT']}"})
elif mode == "random":
  routing["balancers"] = [{"tag": "DECODO", "selector": ["US_ISP_"], "strategy": {"type": "random"}}]
  rules.append({"type": "field", "network": "tcp,udp", "balancerTag": "DECODO"})
else:
  rules.append({"type": "field", "network": "tcp,udp", "outboundTag": "DIRECT"})
print(json.dumps({"log": {"loglevel": "warning"}, "inbounds": [inbound],
                  "outbounds": outbounds, "routing": routing}, indent=2))
PY
install -m 600 /tmp/xray_config.json $X && rm -f /tmp/xray_config.json
marzban restart -n
for i in $(seq 30); do ss -tln | grep -q '127.0.0.1:8444 ' && { echo XRAY_OK; break; }; sleep 2; done
curl -s -o /dev/null -w 'panel=%{http_code}\n' http://127.0.0.1:8000/dashboard/
```

**验证**：

- 输出 `XRAY_OK`；
- 输出 `panel=200`。

如果失败，用 `docker logs --tail 80 marzban-marzban-1` 查看日志。

## 第 7 步：sing-box SNI 分流（占用 443）

```bash
set -euo pipefail; . /etc/proxy-deploy/deploy.env
install -d /etc/sing-box
[ -f /etc/sing-box/config.json ] && cp /etc/sing-box/config.json /etc/sing-box/config.json.bak-$(date +%s)
cat > /etc/sing-box/config.json <<EOF
{
  "log": { "level": "warn", "timestamp": true },
  "inbounds": [ { "type": "direct", "tag": "sni-router", "listen": "0.0.0.0", "listen_port": 443 } ],
  "outbounds": [ { "type": "direct", "tag": "direct" } ],
  "route": { "rules": [
    { "inbound": ["sni-router"], "action": "sniff", "timeout": "1s" },
    { "inbound": ["sni-router"], "protocol": "tls", "domain_suffix": ["$DOMAIN"],
      "action": "route", "outbound": "direct", "override_address": "127.0.0.1", "override_port": 8443 },
    { "inbound": ["sni-router"],
      "action": "route", "outbound": "direct", "override_address": "127.0.0.1", "override_port": 8444 }
  ] }
}
EOF
cat > /etc/systemd/system/sing-box.service <<'EOF'
[Unit]
Description=sing-box SNI router (443 -> Caddy / REALITY)
After=network-online.target docker.service
Wants=network-online.target
[Service]
ExecStart=/usr/local/bin/sing-box run -c /etc/sing-box/config.json
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
[Install]
WantedBy=multi-user.target
EOF
sing-box check -c /etc/sing-box/config.json
systemctl daemon-reload && systemctl enable --now sing-box
sleep 2
ss -tlnp | grep ':443 ' | grep -q sing-box && echo SNI_OK
curl -s -o /dev/null -w 'via8443=%{http_code}\n' --resolve "$DOMAIN:8443:127.0.0.1" "https://$DOMAIN:8443/dashboard/"
curl -s -o /dev/null -w 'via443=%{http_code}\n'  --resolve "$DOMAIN:443:127.0.0.1"  "https://$DOMAIN/dashboard/"
```

**验证**：

- 输出 `SNI_OK`；
- 输出 `via8443=200`；
- `via443` 仅供参考，不是 200 也不影响代理功能。

## 第 8 步：通过 API 设置 Host 并创建第一个用户

```bash
set -euo pipefail; set -a; . /etc/proxy-deploy/deploy.env; set +a
API=http://127.0.0.1:8000/api
TOKEN=$(curl -fsS -X POST $API/admin/token \
  --data-urlencode "username=$MARZBAN_ADMIN_USER" --data-urlencode "password=$MARZBAN_ADMIN_PASS" | jq -r .access_token)
[ -n "$TOKEN" ] && [ "$TOKEN" != null ] || { echo TOKEN_FAIL; exit 1; }
curl -fsS -X PUT $API/hosts -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg a "$VPS_IP" --arg sni "$REALITY_DEST" \
        '{VLESS_REALITY_INBOUND: [{remark: "REALITY ({USERNAME})", address: $a, port: 443, sni: $sni, fingerprint: "chrome"}]}')" \
  | jq -c '.VLESS_REALITY_INBOUND[0] | {address, port, sni, fingerprint}'
curl -fsS -X POST $API/user -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg u "$FIRST_USER" \
        '{username: $u, status: "active", proxies: {vless: {flow: "xtls-rprx-vision"}}, inbounds: {vless: ["VLESS_REALITY_INBOUND"]}}')" \
  > /etc/proxy-deploy/first-user.json
chmod 600 /etc/proxy-deploy/first-user.json
jq -r '"user=" + .username + " status=" + .status + " has_sub=" + ((.subscription_url | length) > 0 | tostring)' /etc/proxy-deploy/first-user.json
```

**验证**：

- Host 那一行显示 `port` 为 443，`sni` 为 `REALITY_DEST`；
- 用户那一行输出 `status=active has_sub=true`。

## 第 9 步：Clash 订阅模板

替换 Marzban 默认的 Clash 订阅格式。订阅链接本身不变，模板只决定 Clash 拉取到的配置内容：只有一个手动选择的策略组 `AutoProxy`，国内 IP 直连，其余流量走代理。已导入的客户端更新订阅后生效。

```bash
set -euo pipefail; . /etc/proxy-deploy/deploy.env
install -d /var/lib/marzban/templates/clash
cat > /var/lib/marzban/templates/clash/default.yml <<'EOF'
mode: rule
{{ conf | except("proxy-groups", "port", "mode", "rules") | yaml }}
proxy-groups:
  - name: "AutoProxy"
    type: select
    proxies:
      {{ proxy_remarks | yaml | indent(6) }}
rules:
  - GEOIP,CN,DIRECT
  - MATCH,AutoProxy
EOF
E=/opt/marzban/.env; cp $E $E.bak-$(date +%s)
sed -i -E '/^[#[:space:]]*(CUSTOM_TEMPLATES_DIRECTORY|CLASH_SUBSCRIPTION_TEMPLATE)[[:space:]]*=/d' $E
printf 'CUSTOM_TEMPLATES_DIRECTORY = "/var/lib/marzban/templates/"\nCLASH_SUBSCRIPTION_TEMPLATE = "clash/default.yml"\n' >> $E
marzban restart -n
for i in $(seq 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/dashboard/)" = 200 ] && break; sleep 2
done
SUB=$(jq -r .subscription_url /etc/proxy-deploy/first-user.json)
umask 077; curl -s --resolve "$DOMAIN:8443:127.0.0.1" -A 'clash-verge/v2.0.0' "$SUB" > /tmp/sub.yml || true
grep -q 'AutoProxy' /tmp/sub.yml && grep -q 'GEOIP,CN,DIRECT' /tmp/sub.yml && echo SUB_TEMPLATE_OK || echo SUB_TEMPLATE_FAIL
rm -f /tmp/sub.yml
```

**验证**：输出 `SUB_TEMPLATE_OK`。订阅内容里有 UUID，只检查关键字，不要打印订阅内容。

## 第 10 步：端到端验收

在 VPS 本机起一个临时客户端，连 `127.0.0.1:443`。这样能覆盖完整的服务端链路：sing-box → REALITY → 出站。

```bash
set -euo pipefail; set -a; . /etc/proxy-deploy/deploy.env; set +a
UUID=$(jq -r .proxies.vless.id /etc/proxy-deploy/first-user.json)
umask 077; cat > /tmp/e2e.json <<EOF
{ "log": { "level": "warn" },
  "inbounds": [ { "type": "mixed", "listen": "127.0.0.1", "listen_port": 10808 } ],
  "outbounds": [ { "type": "vless", "server": "127.0.0.1", "server_port": 443, "uuid": "$UUID", "flow": "xtls-rprx-vision",
    "tls": { "enabled": true, "server_name": "$REALITY_DEST", "utls": { "enabled": true, "fingerprint": "chrome" },
             "reality": { "enabled": true, "public_key": "$REALITY_PUBLIC_KEY", "short_id": "$REALITY_SHORT_ID" } } } ] }
EOF
sing-box run -c /tmp/e2e.json >/tmp/e2e.log 2>&1 & P=$!; sleep 2
for i in 1 2 3; do echo "exit_ip=$(curl -s --max-time 20 -x socks5h://127.0.0.1:10808 https://api.ipify.org)"; done
curl -s -o /dev/null --max-time 60 -x socks5h://127.0.0.1:10808 -w 'download=%{speed_download}B/s\n' \
  'https://speed.cloudflare.com/__down?bytes=10000000'
kill $P; rm -f /tmp/e2e.json
```

**验证**：三次 `exit_ip` 都要有值。不同模式下应该看到：

- `fixed`：三次相同，而且不等于 `VPS_IP`；
- `random`：可能不同，都不等于 `VPS_IP`；
- `direct`：等于 `VPS_IP`。

拿不到 IP 时，查看 `/tmp/e2e.log` 和 `docker logs --tail 80 marzban-marzban-1`。

## 第 11 步：Claude 镜像的出口

第 11–15 步部署 Claude 网页版镜像，用的是 [reclaude](https://github.com/adryfish/reclaude)。它把 claude.ai 网页端镜像到 `https://claude.<DOMAIN>`，浏览器登录一次后即可直接使用，出口固定走 `FIXED_PORT`。

- **子域名**：
  - `claude` 是主站；
  - `content` 用于 artifact 预览；
  - `claude-mcp` 用于 visualize 等 MCP 应用。

  浏览器经 443 访问这三个子域名，sing-box 按 SNI 把它们转给 Caddy，安全组不需要改动。
- **在用环境的差异**：镜像部署在另一台 VPS 上，由 Caddy 直接监听 443。reclaude 的配置和这里相同。
- `[HUMAN]` **使用风险**：这类镜像可能违反 Anthropic 服务条款，账号风险由使用者自行承担，开始前须由操作人员确认知悉。

本步骤给镜像单独开一个出口入口。`EXIT_MODE=direct` 时没有 decodo 出站，改由 VPS 直接出站。

```bash
set -euo pipefail; F=/etc/proxy-deploy/deploy.env; . $F
X=/var/lib/marzban/xray_config.json
jq -e '.inbounds[] | select(.tag == "mirror_socks")' $X >/dev/null && { echo "mirror_socks 已存在，停止"; exit 1; }
grep -q '^MIRROR_SOCKS_PASS=' $F || echo "MIRROR_SOCKS_PASS=$(openssl rand -hex 12)" >> $F; . $F
OUT=$(jq -r --arg o "US_ISP_${FIXED_PORT:-}" 'if ([.outbounds[].tag] | index($o)) then $o else "DIRECT" end' $X)
echo "mirror egress -> $OUT"
cp $X $X.bak-$(date +%s)
jq --arg p "$MIRROR_SOCKS_PASS" --arg out "$OUT" '
  .inbounds += [{tag: "mirror_socks", listen: "172.17.0.1", port: 2082, protocol: "socks",
                 settings: {auth: "password", accounts: [{user: "mirror", pass: $p}], udp: false}}]
  | .routing.rules = [.routing.rules[0], {type: "field", inboundTag: ["mirror_socks"], outboundTag: $out}] + .routing.rules[1:]
' $X > /tmp/x.json && install -m 600 /tmp/x.json $X && rm /tmp/x.json
marzban restart -n
for i in $(seq 30); do ss -tln | grep -q '172.17.0.1:2082 ' && { echo MIRROR_SOCKS_OK; break; }; sleep 2; done
for i in 1 2 3; do echo "mirror_exit=$(curl -s --max-time 20 -x "socks5h://mirror:$MIRROR_SOCKS_PASS@172.17.0.1:2082" https://api.ipify.org)"; done
```

**验证**：

- 输出 `MIRROR_SOCKS_OK`；
- 三次 `mirror_exit` 相同；
- `mirror_exit` 是 `FIXED_PORT` 对应的 IP，即资源准备 C 第 4 条测出的那个；`direct` 模式下等于 `VPS_IP`。

## 第 12 步：部署 reclaude

镜像版本固定为 2.17.1，第 13 步的修复依赖这个版本的行为。

```bash
set -euo pipefail; F=/etc/proxy-deploy/deploy.env; . $F
[ -d /opt/reclaude ] && { echo "/opt/reclaude 已存在，停止"; exit 1; }
ss -tln | grep -E '127.0.0.1:(4000|4001|4010|4020) ' && { echo PORTS_BUSY; exit 1; }
grep -q '^RECLAUDE_SECRET=' $F || echo "RECLAUDE_SECRET=$(openssl rand -hex 32)" >> $F
grep -q '^MIRROR_ADMIN_PASS=' $F || echo "MIRROR_ADMIN_PASS=$(openssl rand -hex 16)" >> $F
. $F
install -d -m 700 /opt/reclaude
cat > /opt/reclaude/docker-compose.yml <<'EOF'
services:
  reclaude:
    image: adryfish/reclaude:2.17.1
    ports:
      - "127.0.0.1:4000:4000"
      - "127.0.0.1:4010:4010"
      - "127.0.0.1:4020:4020"
    env_file:
      - .env
    volumes:
      - ./data:/data
    restart: unless-stopped
EOF
(umask 077; cat > /opt/reclaude/.env <<EOF
TZ=America/Los_Angeles
ALL_PROXY=socks5://mirror:$MIRROR_SOCKS_PASS@172.17.0.1:2082
RECLAUDE_SECRET=$RECLAUDE_SECRET
SERVER_PORT=4000
CONTENT_SERVER_PORT=4010
MAIN_SERVER_URL=https://claude.$DOMAIN
CONTENT_SERVER_URL=https://content.$DOMAIN
RECLAUDE_ENABLE_CONTENT_SERVER=true
RECLAUDE_ENABLE_MCP_SERVER=true
RECLAUDE_MCP_SERVER_URL=https://claude-mcp.$DOMAIN
RECLAUDE_FEATURE_LEVEL=full
ADMIN_USERNAME=admin
ADMIN_PASSWORD=$MIRROR_ADMIN_PASS
EOF
)
cd /opt/reclaude && docker compose up -d
for i in $(seq 30); do ss -tln | grep -q '127.0.0.1:4020 ' && break; sleep 2; done
docker ps --format '{{.Names}} {{.Status}}' | grep reclaude-reclaude-1
for p in 4000 4010 4020; do ss -tln | grep -q "127.0.0.1:$p " && echo "port_$p=up" || echo "port_$p=down"; done
```

**验证**：

- 能看到 `reclaude-reclaude-1 Up ...`；
- 三个端口都是 `up`。

## 第 13 步：修复 visualize

**原因**：claude.ai 前端根据配置项 `mcpAppsSandboxProxyUrl` 推算 visualize 沙箱的 origin，并严格校验沙箱发来的消息来源。reclaude 把沙箱 iframe 改到了镜像域名下，两者不一致，消息被丢弃，页面显示 "There was a problem displaying content from visualize."。

**修法**：这个代理只处理 `*.js` 请求，把该配置项改成镜像的 MCP 地址，其余内容原样透传。代理不可用时，Caddy 会退回直连 reclaude。

```bash
set -euo pipefail; . /etc/proxy-deploy/deploy.env
install -d /opt/reclaude-rewrite
cat > /opt/reclaude-rewrite/rewrite_proxy.py <<'PY'
#!/usr/bin/env python3
"""Rewrite proxy in front of reclaude for *.js: point mcpAppsSandboxProxyUrl at the mirror's MCP server."""
import gzip, http.client, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "4001"))
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "4000"))
MCP_SERVER_URL = os.environ["MCP_SERVER_URL"].rstrip("/")
REPLACEMENTS = [(b'"https://sandbox.claudemcpcontent.com/mcp_apps"',
                 f'"{MCP_SERVER_URL}/mcp/sandbox/mcp_apps"'.encode())]
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade"}
_lock = threading.Lock()

def log(msg):
    with _lock:
        print(msg, file=sys.stderr, flush=True)

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        path = self.path.split("?")[0]
        try:
            conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM_PORT, timeout=120)
            conn.putrequest(self.command, self.path, skip_host=True, skip_accept_encoding=True)
            for k, v in self.headers.items():
                if k.lower() not in HOP_BY_HOP and k.lower() != "accept-encoding":
                    conn.putheader(k, v)
            conn.putheader("Accept-Encoding", "identity")
            conn.endheaders(body)
            resp = conn.getresponse()
            data = resp.read()
            conn.close()
        except Exception as e:
            log(f"upstream error {self.command} {path}: {e!r}")
            self.send_error(502)
            return
        ctype = (resp.getheader("Content-Type") or "").lower()
        enc = (resp.getheader("Content-Encoding") or "").lower()
        changed = 0
        if resp.status == 200 and "javascript" in ctype and enc in ("", "identity", "gzip") and data:
            try:
                raw = gzip.decompress(data) if enc == "gzip" else data
                for old, new in REPLACEMENTS:
                    n = raw.count(old)
                    if n:
                        raw, changed = raw.replace(old, new), changed + n
                if changed:
                    data = raw
                    log(f"rewrote {changed}x in {path}")
            except Exception as e:
                log(f"rewrite skipped for {path}: {e!r}")
                changed = 0
        self.send_response_only(resp.status, resp.reason)
        for k, v in resp.getheaders():
            lk = k.lower()
            if lk in HOP_BY_HOP or lk == "content-length" or (changed and lk in ("content-encoding", "etag")):
                continue
            self.send_header(k, v)
        if self.command == "HEAD":
            if resp.getheader("Content-Length") is not None:
                self.send_header("Content-Length", resp.getheader("Content-Length"))
            self.end_headers()
            return
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if resp.status not in (204, 304):
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _proxy

class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True

if __name__ == "__main__":
    log(f"reclaude-rewrite 127.0.0.1:{LISTEN_PORT} -> 127.0.0.1:{UPSTREAM_PORT}, sandbox -> {MCP_SERVER_URL}/mcp/sandbox/mcp_apps")
    Server(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
PY
chmod 755 /opt/reclaude-rewrite/rewrite_proxy.py
cat > /etc/systemd/system/reclaude-rewrite.service <<EOF
[Unit]
Description=Rewrite proxy for reclaude *.js (visualize sandbox origin)
After=network.target docker.service

[Service]
Environment=LISTEN_PORT=4001 UPSTREAM_PORT=4000 MCP_SERVER_URL=https://claude-mcp.$DOMAIN
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=/usr/bin/python3 /opt/reclaude-rewrite/rewrite_proxy.py
DynamicUser=yes
Restart=always
RestartSec=2
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now reclaude-rewrite
sleep 1; systemctl is-active reclaude-rewrite
curl -s -o /dev/null -w 'direct=%{http_code} ' -H "Host: claude.$DOMAIN" http://127.0.0.1:4000/x.js
curl -s -o /dev/null -w 'via_rewrite=%{http_code}\n' -H "Host: claude.$DOMAIN" http://127.0.0.1:4001/x.js
```

**验证**：

- 输出 `active`；
- `direct` 和 `via_rewrite` 的状态码相同，未登录时都是 302。

## 第 14 步：镜像站点（Caddy）

```bash
set -euo pipefail; . /etc/proxy-deploy/deploy.env
C=/etc/caddy/Caddyfile
grep -q "claude.$DOMAIN" $C && { echo "Caddyfile 里已有镜像站点，停止"; exit 1; }
cp $C $C.bak-$(date +%s)
cat >> $C <<EOF

claude.$DOMAIN:8443 {
    @js path *.js
    handle @js {
        encode zstd gzip
        reverse_proxy 127.0.0.1:4001 127.0.0.1:4000 {
            lb_policy first
            lb_try_duration 3s
            fail_duration 30s
        }
    }
    handle {
        reverse_proxy 127.0.0.1:4000
    }
}

content.$DOMAIN:8443 {
    reverse_proxy 127.0.0.1:4010
}

claude-mcp.$DOMAIN:8443 {
    reverse_proxy 127.0.0.1:4020
}
EOF
caddy validate --config $C --adapter caddyfile 2>&1 | tail -1
systemctl reload caddy
for h in claude content claude-mcp; do
  for i in $(seq 24); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 --resolve "$h.$DOMAIN:443:127.0.0.1" "https://$h.$DOMAIN/" || true)
    [ "$code" != 000 ] && break; sleep 5
  done
  echo "$h=$code"
done
curl -s -o /dev/null -w 'login=%{http_code}\n'   --resolve "claude.$DOMAIN:443:127.0.0.1"     "https://claude.$DOMAIN/login"
curl -s -o /dev/null -w 'sandbox=%{http_code}\n' --resolve "claude-mcp.$DOMAIN:443:127.0.0.1" "https://claude-mcp.$DOMAIN/mcp/sandbox/mcp_apps"
```

**验证**：

- 输出 `Valid configuration`；
- 三个子域名的状态码都不是 `000`（证书已签发，经 443 可以访问）；
- 输出 `login=200`；
- 输出 `sandbox=200`。这个请求经 reclaude 和 `mirror_socks` 出口访问上游，能同时验证容器的出站链路。

## 第 15 步：登录镜像并验收

先由操作人员完成登录 `[HUMAN]`：

1. 在浏览器登录 claude.ai，从 Cookie 中复制 `sessionKey`（以 `sk-ant-sid01-` 开头）。建议这个浏览器也经本套代理、走同一个固定出口 IP，避免同一会话出现在不同 IP 上。
2. 打开 `https://claude.<DOMAIN>/login`，在 session key 输入框填入后提交。sessionKey 等同于账号凭据，只由操作人员处理，不要交给 agent，也不要写进任何文件。
3. 在镜像里让 Claude 画一张图表，确认 visualize 可以显示。

然后由 agent 检查日志：

```bash
journalctl -u reclaude-rewrite --no-pager -o cat | grep -m3 rewrote
docker logs reclaude-reclaude-1 2>&1 | grep -c 'MCP GET https://sandbox.claudemcpcontent.com'
```

**验证**：

- 至少有一行 `rewrote 1x in ...`；
- 计数大于 0。

## 第 16 步：安全加固

**fail2ban**：agent 直接执行。

```bash
set -euo pipefail
apt-get install -y fail2ban
cat > /etc/fail2ban/jail.d/sshd.local <<'EOF'
[sshd]
enabled = true
maxretry = 5
bantime = 1d
EOF
systemctl restart fail2ban && fail2ban-client status sshd | head -3
```

**SSH 只允许密钥登录**：`[HUMAN]`。agent 当前可能正在用密码认证登录，直接修改会导致无法重新登录。须由操作人员确认密钥登录可用后再执行：

```bash
cat > /etc/ssh/sshd_config.d/90-hardening.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
EOF
sshd -t && systemctl reload ssh
```

## 第 17 步：交付汇报

部署完成后，向操作人员提交以下信息：

- **面板地址**：`https://<DOMAIN>:8443/dashboard/`，管理员用户名是 `admin`。密码不要输出，提示操作人员用 `sudo grep MARZBAN_ADMIN_PASS /etc/proxy-deploy/deploy.env` 查看。
- **第一个用户的订阅链接**：`sudo jq -r .subscription_url /etc/proxy-deploy/first-user.json` 的输出。可以直接写进汇报，用于导入 Clash Verge Rev。
- **出口模式和第 10 步测到的出口 IP。**
- **Claude 镜像**：地址 `https://claude.<DOMAIN>`，第 11 步测到的镜像出口 IP，第 15 步的验收结果。
- **跳过或失败的步骤**，附上相关输出。
- **秘密存放位置**：`/etc/proxy-deploy/deploy.env`、`/etc/proxy-deploy/first-user.json` 和 `/opt/reclaude/.env`，权限都是 600。

## 第 18 步：客户端配置 `[HUMAN]`

本步骤在操作人员自己的电脑上完成，不在 VPS 上执行。

**Clash Verge Rev**：
1. 在「订阅」页粘贴第 17 步交付的订阅链接，然后导入；
2. 在「代理」页的 `AutoProxy` 组里选中节点，名称以 `REALITY` 开头；
3. 在「设置 → 端口设置」查看混合代理端口，默认是 7897。下文按 7897 书写，如有不同请替换。

**终端（Claude Code）**：

终端程序不会自动使用 Clash 的系统代理，需要用环境变量指定。下面的函数有两个作用：
- **只影响 `claude` 命令**：只给它加代理，其他命令照常运行；
- **代理不通时直接报错**：Clash 没有运行时，`claude` 会连接失败，而不会绕过代理、用本机 IP 直连。

把函数加到 `~/.bashrc` 里；如果用的是 zsh（macOS 默认），就加到 `~/.zshrc`：

```bash
# Claude Code 经 Clash Verge Rev 的本地端口出站
unalias claude 2>/dev/null
function claude {
  local p=http://127.0.0.1:7897
  HTTPS_PROXY=$p HTTP_PROXY=$p https_proxy=$p http_proxy=$p NO_PROXY=localhost,127.0.0.1 command claude "$@"
}
```

然后执行 `source ~/.bashrc`（或 `source ~/.zshrc`）。Windows 可以在 WSL 里按同样方式配置；如果用 PowerShell，先执行 `$env:HTTPS_PROXY='http://127.0.0.1:7897'` 再运行 `claude`。

首次运行 `claude` 登录时，浏览器也要开着 Clash 的系统代理，这样登录和之后的使用都来自同一个出口 IP。

**验证**：

```bash
curl -s -x http://127.0.0.1:7897 https://api.ipify.org; echo   # 应等于第 10 步的出口 IP
type claude | head -1                                            # 应显示 claude 是 function
```

`EXIT_MODE=random` 时，出口 IP 会在几个 IP 之间切换，Claude 账号的访问来源会频繁变化。使用 Claude Code 时，`EXIT_MODE` 应设为 `fixed`。

## Claude 镜像维护

- **账号会话**：保存在 `/opt/reclaude/data/reclaude.db`，由 `RECLAUDE_SECRET` 加密。修改这个 secret 或删除 `data/` 都会让会话失效，需要重新登录。升级或迁移前，先备份 `data/` 和 `.env`。
- **reclaude 版本**：固定在 2.17.1。第 13 步依赖它对 iframe 地址的改写方式（`*.claudemcpcontent.com` 改到 `<RECLAUDE_MCP_SERVER_URL>/mcp/<子域>/`），升级后要重新做第 15 步的验收。
- **visualize 再次报错时**：如果 `journalctl -u reclaude-rewrite` 里不再出现 `rewrote`，说明 claude.ai 改了 `mcpAppsSandboxProxyUrl` 这个配置常量，需要重新分析。
- **共享给他人使用时**：把 `RECLAUDE_FEATURE_LEVEL` 改为 `standard`，禁用账单、订阅等敏感页面。

## 可选步骤

**A. Claude 中转：部署 [claude-relay-service](https://github.com/Wei-Shaw/claude-relay-service)**。分三步：

1. **部署**。按 README 的 Docker 方式，在 `/opt` 下执行 `curl -fsSL https://pincc.ai/crs-compose.sh -o crs-compose.sh`。先读一遍脚本再运行，它可能会交互提问，不确定的选项询问操作人员。运行完再执行 `docker compose up -d`。在 `.env` 或环境变量里设 `BIND_HOST=127.0.0.1`，只监听本机。
2. **给中转单独开一个出口入口**，固定走一个 decodo 端口。容器访问不到宿主机的 127.0.0.1，所以这个入口监听在 docker 网桥地址上：

```bash
set -euo pipefail; . /etc/proxy-deploy/deploy.env
RELAY_PASS=$(openssl rand -hex 12); echo "RELAY_SOCKS_PASS=$RELAY_PASS" >> /etc/proxy-deploy/deploy.env
X=/var/lib/marzban/xray_config.json; cp $X $X.bak-$(date +%s)
jq --arg p "$RELAY_PASS" --arg out "US_ISP_$FIXED_PORT" '
  .inbounds += [{tag: "relay_socks", listen: "172.17.0.1", port: 2081, protocol: "socks",
                 settings: {auth: "password", accounts: [{user: "relay", pass: $p}], udp: false}}]
  | .routing.rules = [.routing.rules[0], {type: "field", inboundTag: ["relay_socks"], outboundTag: $out}] + .routing.rules[1:]
' $X > /tmp/x.json && install -m 600 /tmp/x.json $X && rm /tmp/x.json
marzban restart -n
```

   然后在中转后台，把每个 Claude 账号的代理填成 `socks5://relay:<RELAY_SOCKS_PASS>@172.17.0.1:2081`，密码在 `deploy.env` 里。
3. **用 HTTPS 访问中转**：

- `[HUMAN]` 给子域名 `relay.<DOMAIN>` 添加 A 记录，指向 `VPS_IP`；
- agent 在 Caddyfile 里追加一段 `relay.<DOMAIN>:8443 { reverse_proxy 127.0.0.1:4567 }`，然后执行 `systemctl reload caddy`。

   sing-box 不用改，它的 `domain_suffix` 规则会把子域名也转给 Caddy。

**B. 内网穿透：部署 [frp](https://github.com/fatedier/frp)**：

- 从 Releases 下载 `frp_<版本>_linux_amd64.tar.gz`，解压到 `/opt/frp`；
- `frps.toml` 写 `bindPort = 7000` 和 `auth.token = "<openssl rand -hex 16>"`；
- 写一个 systemd 服务运行 `/opt/frp/frps -c /opt/frp/frps.toml`；
- `[HUMAN]` 在安全组放行 7000。映射出来的端口只放行指定来源 IP，或者改用 `stcp` 类型，这样就不需要开放公网端口。

## 故障排查

| 现象                                                                           | 检查                                                                                                                                                                         |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 第 4 步没有`CERT_OK`                                                         | `journalctl -u caddy -n 50`；安全组是否放行 TCP 80；`DOMAIN` 是否解析到 `VPS_IP`                                                                                       |
| 第 6 步没有`XRAY_OK`                                                         | `docker logs --tail 80 marzban-marzban-1`，常见原因是 `xray_config.json` 格式错误或私钥为空                                                                              |
| 第 7 步 443 端口被占用                                                         | `ss -tlnp \| grep ':443 '`，汇报给操作人员，不要终止进程                                                                                                                    |
| 第 8 步`TOKEN_FAIL`                                                          | `.env` 里的 `SUDO_USERNAME` 和 `SUDO_PASSWORD` 是否生效（需要执行过 `marzban restart -n`）                                                                           |
| 第 9 步`SUB_TEMPLATE_FAIL` | `/opt/marzban/.env` 里的 `CUSTOM_TEMPLATES_DIRECTORY`、`CLASH_SUBSCRIPTION_TEMPLATE` 是否各只有一行；`docker logs --tail 50 marzban-marzban-1` 有无模板渲染错误 |
| 第 10 步拿不到出口 IP                                                           | `/tmp/e2e.log`；`EXIT_MODE` 是 `fixed` 或 `random` 时，在 VPS 上执行 `curl -x socks5h://<user>:<pass>@isp.decodo.com:<port> https://api.ipify.org` 单独检查 decodo |
| 客户端连不上，但第 10 步通过                                                    | 问题在 VPS 外部：安全组是否放行 TCP 443；客户端的订阅是否已更新                                                                                                              |
| 第 14 步的子域名状态码是`000`                                            | 子域名的 DNS 记录是否生效；`journalctl -u caddy -n 50` 中证书签发是否报错                                                                                                  |
| 第 14 步的`sandbox` 不是 200                                             | `docker logs --tail 50 reclaude-reclaude-1`；重跑第 11 步末尾的 `mirror_exit` 检查，确认出口可用                                                                             |
| 镜像里 visualize 显示 "There was a problem displaying content from visualize." | 浏览器开发者工具勾选 Disable cache 后刷新，绕过缓存的旧 JS（不影响登录状态）；仍然不行时按「Claude 镜像维护」检查`rewrote` 日志                                                           |
