from django import forms
from .models import * # Keep your existing imports

class VideoUploadForm(forms.Form):
    # We update the 'accept' attribute to allow video/* AND image/*
    sequence_length = forms.IntegerField(label="Check how many frames?")
    upload_video_file = forms.FileField(
        label="Select Video or Image",
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'video/*,image/*'  # <--- THIS IS THE FIX
        })
    )