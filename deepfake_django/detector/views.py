from PIL import Image

from django.shortcuts import render

from .forms import ImageUploadForm
from . import ml_model


def index(request):
    context = {"form": ImageUploadForm()}

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        context["form"] = form

        if form.is_valid():
            uploaded = form.cleaned_data["image"]
            try:
                image = Image.open(uploaded)
                image.load()  # force-read now, while the file handle is still open
            except Exception:
                form.add_error("image", "That file isn't a readable image. Upload a JPG or PNG.")
            else:
                try:
                    real_prob, fake_prob = ml_model.predict(image)
                except RuntimeError as exc:
                    context["model_error"] = str(exc)
                else:
                    context["result"] = {
                        "verdict": "manipulated" if fake_prob >= 0.5 else "authentic",
                        "real_prob": real_prob,
                        "fake_prob": fake_prob,
                        "real_pct": round(real_prob * 100, 1),
                        "fake_pct": round(fake_prob * 100, 1),
                    }
                    # Re-show the same image the user uploaded, without saving it to disk.
                    uploaded.seek(0)
                    context["preview_data_uri"] = _to_data_uri(uploaded)

    _, _, val_f1, val_auc = ml_model.get_model()
    context["val_f1"] = val_f1
    context["val_auc"] = val_auc
    if ml_model.get_load_error():
        context["model_error"] = ml_model.get_load_error()

    return render(request, "detector/index.html", context)


def _to_data_uri(uploaded_file):
    import base64
    content_type = uploaded_file.content_type or "image/jpeg"
    encoded = base64.b64encode(uploaded_file.read()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"
