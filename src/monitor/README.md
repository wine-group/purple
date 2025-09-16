The PURPLE-AIMD can be started using the adapter script, with the following configuration parameters as an example (update to match your radio password, hostnames, etc):

python3 purple_aimd_adapter.py \
  --radio-host 10.0.10.247 \
  --radio-web-user ubnt \
  --radio-web-password "u(a)=UP~0OBu" \
  --radio-ssh-user ubnt \
  --radio-ssh-password "u(a)=UP~0OBu" \
  --radio-interface ath0 \
  --gateway-host 10.0.10.250 \
  --gateway-user wine \
  --gateway-key /home/wine/auth.txt \
  --gateway-interface eth0 \
  --target 5.0 \
  --min-rate 50.0 \
  --max-rate 120.0 \
  --initial-rate 100.0

