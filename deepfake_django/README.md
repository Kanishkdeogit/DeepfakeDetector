# Deepfake Face Detector — Django App

A Django web app around the trained `CustomDeepfakeCNN` checkpoint. Upload a
face image; it returns Authentic / Manipulated with a confidence breakdown.

Tested end-to-end with the actual trained checkpoint before delivery —
GET page load, POST image upload + inference, and invalid-file error handling
all verified working.

## 1. Project layout

```
deepfake_django/
├── manage.py
├── requirements.txt
├── deepfake_site/          # project settings/urls
└── detector/                # the app
    ├── ml_model.py           # model architecture + cached loader
    ├── forms.py
    ├── views.py
    ├── urls.py
    ├── templates/detector/index.html
    └── static/detector/style.css
```

## 2. Add your checkpoint

Place `custom_cnn_deepfake_detector_final.pth` in the **project root**
(same folder as `manage.py`). The app looks for it there by default.

To use a different location, set an environment variable instead of moving the file:

```bash
export DEEPFAKE_MODEL_PATH=/path/to/your/checkpoint.pth
```

## 3. Run locally

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

The model loads once per process (cached in `ml_model.py`), not on every
request, so the first prediction after startup is slightly slower than
subsequent ones.

## 4. Deploy

Any standard Django hosting works (Railway, Render, PythonAnywhere, a VPS with
gunicorn + nginx, etc.). Three things to change for production:

1. In `deepfake_site/settings.py`, set `DEBUG = False` and add your domain to
   `ALLOWED_HOSTS`.
2. Run `python manage.py collectstatic` so `style.css` is served correctly
   outside of Django's dev-only static handling.
3. Use a real WSGI/ASGI server instead of `runserver`, e.g.:
   ```bash
   pip install gunicorn
   gunicorn deepfake_site.wsgi:application
   ```

If your host has a small free-tier disk/RAM limit, note the checkpoint is
~55MB and PyTorch itself is a fairly large dependency — pick a host/tier that
comfortably fits both (e.g. Render's free web service tier works, Railway's
free tier works, very small 512MB-RAM tiers may struggle).

## Notes

- Inference uses the same `Resize(224,224)` + ImageNet normalization as the
  training notebook's validation transform — no augmentation at inference time.
- `dropout_rate` is read from the checkpoint's saved `best_hyperparameters`,
  so the architecture is reconstructed exactly as trained — no hardcoding.
- This is a research/portfolio demo, not a certified forensic tool. Add a
  visible disclaimer if you deploy it publicly (one is already included in
  the page footer).
