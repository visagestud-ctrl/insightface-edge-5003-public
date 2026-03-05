# Render deploy (InsightFace + Edge)

## 1) Upload folder to GitHub
Upload this folder as `render-insightface-edge` inside your repo.

## 2) Create service in Render
- New + -> Blueprint
- Select repo
- Render reads `render.yaml` and creates service `insightface-edge-5003`

## 3) Wait for build
First build can take long (model deps).

## 4) Test
- `https://<your-render-domain>/health`
- `https://<your-render-domain>/detect` (POST multipart `image`, optional `edge_strength`)

## 5) Connect frontend
In `side_points_from_points.html`, set endpoint to:
`https://<your-render-domain>/detect`

## Notes
- Free plan can sleep after inactivity.
- Cold start can be slow.
