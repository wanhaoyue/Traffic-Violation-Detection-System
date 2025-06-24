# detection/models.py

from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class ViolationRecord(models.Model):
    violation_type       = models.CharField(max_length=100)
    license_plate_number = models.CharField(max_length=50, null=True, blank=True)
    plate_image = models.ImageField(upload_to='plates/', null=True, blank=True)
    detection_time       = models.DateTimeField(auto_now_add=True)
    original_image       = models.ImageField(upload_to='violations/original/')
    detected_image       = models.ImageField(upload_to='violations/detected/')

    # ─── NEW FIELDS ───────────────────────────────────────────
    location         = models.CharField(max_length=100, null=True, blank=True)
    incident_date    = models.DateField(null=True, blank=True)
    incident_time    = models.TimeField(null=True, blank=True)
    location_details = models.TextField(null=True, blank=True)
    # ──────────────────────────────────────────────────────────

    def __str__(self):
        return (
            f"{self.violation_type} - "
            f"{self.license_plate_number or 'Unknown'} @ "
            f"{self.incident_date} {self.incident_time}"
        )
