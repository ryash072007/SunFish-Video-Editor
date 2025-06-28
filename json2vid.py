import requests, json, time

API_KEY = 'Lrb0qJF3qsJGvE9bEzLHtt5fC4bgvkgSSE91NElI'
HEADERS = {'Content-Type': 'application/json', 'x-api-key': API_KEY}
RENDER_URL = 'https://api.shotstack.io/stage/render'
INGEST_URL = 'https://api.shotstack.io/ingest/stage/upload'

def upload_video_and_render(video_path, json_path):
    # Request a signed S3 upload URL
    resp = requests.post(INGEST_URL, headers=HEADERS)
    if not resp.ok:
        raise Exception(f"Failed to get upload URL: {resp.text}")
    data = resp.json()['data']
    signed_url = data['attributes']['url']
    source_id = data['id']
    print(f"[1/4] Upload URL received, source ID = {source_id}")

    # Upload the video file to S3
    with open(video_path, 'rb') as f:
        put = requests.put(signed_url, data=f)
        if put.status_code != 200:
            raise Exception(f"S3 upload failed: {put.text}")
    print("[2/4] Video uploaded to S3")

    # Load and update the JSON template
    with open(json_path) as f:
        payload = json.load(f)

    def patch(obj):
        if isinstance(obj, dict):
            # Replace placeholder with the signed URL
            if obj.get('src') == "{{VIDEO_URL}}":
                return {**obj, 'src': signed_url}
            return {k: patch(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [patch(v) for v in obj]
        elif isinstance(obj, str):
            return obj.replace("{{VIDEO_URL}}", signed_url)
        return obj

    payload = patch(payload)
    print("[3/4] JSON template updated with video URL")

    # Submit the render request
    render_resp = requests.post(RENDER_URL,
                                headers={**HEADERS, 'Accept': 'application/json'},
                                json=payload)
    if render_resp.status_code not in (200, 201):
        raise Exception(f"Render failed: {render_resp.text}")
    render_id = render_resp.json()['response']['id']
    print(f"[4/4] Render queued. ID = {render_id}")

    # Check rendering status until completion
    print("Polling for status updates...")
    while True:
        r = requests.get(f"{RENDER_URL}/{render_id}",
                         headers={**HEADERS, 'Accept': 'application/json'})
        resp = r.json()['response']
        print("Status:", resp.get('status'))
        if resp.get('status') in ('done', 'failed'):
            print("Render complete. URL:", resp.get('url'))
            break
        time.sleep(5)

# Run it
upload_video_and_render("video.mp4", "shotstack_full.json")
