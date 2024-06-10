iptables -I INPUT -p tcp -m tcp --dport 5007 -j ACCEPT
service iptables start

# Not sure if it works
# ufw enable
# ufw allow 5007/tcp
