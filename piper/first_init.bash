#!/usr/bin/env bash
# 在宿主机执行：sudo bash piper/first_init.bash
set -e

# 当前 USB-CAN 的序列号；更换适配器时修改这里。
CAN_SERIAL="0036003A5246571120393733"
CAN_NAME="can_piper"

mkdir -p /etc/systemd/network

cat > /etc/systemd/network/10-piper.link <<EOF
[Match]
Driver=gs_usb
Property=ID_SERIAL_SHORT=${CAN_SERIAL}

[Link]
NamePolicy=
Name=${CAN_NAME}
EOF

cat > /etc/systemd/network/10-piper.network <<EOF
[Match]
Name=${CAN_NAME}

[CAN]
BitRate=1M
EOF

systemctl enable --now systemd-networkd
networkctl reload
udevadm control --reload-rules

echo "配置完成。停止机械臂程序并拔插 USB-CAN，使名称和自动启动配置生效。"
echo "检查：ip -details link show ${CAN_NAME}"
