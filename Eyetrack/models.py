# models.py
from django.db import models

class GazeTrackingResult(models.Model):
    #image = models.ImageField(upload_to='gaze_images/')
    encoded_image = models.TextField()
    feedback = models.TextField(default="No feedback provided.") 
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"GazeTrackingResult #{self.id}"


