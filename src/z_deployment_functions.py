from utils.utils_envvar import EnvVar, GeneralUtils
import requests

env_vars = EnvVar()

def get_ngrok_tunnel_url():
    tunnel_info = GeneralUtils.open_json_to_dict(env_vars.url_tunnel_path)

    tunnel_count = len(tunnel_info["tunnels"])

    if tunnel_count<1:
        print("more than 1 http tunnel, script might not work please go to src/z_development_functions.py to debug")

    # Assumes that it is the first tunnel there, conisder other ways to make this more robust
    url = tunnel_info["tunnels"][0]["public_url"]

    return url

def set_telegram_webhook():
    ngrok_url = get_ngrok_tunnel_url()
    webhook_url = f"https://api.telegram.org/bot{env_vars.tele_api_key}/setWebhook?url={ngrok_url}/"
    x = requests.get(webhook_url, timeout = 10)
    print(f"Webhook status: {x}")


if __name__ == '__main__':
    set_telegram_webhook()