from django.db import models
from django.db.models.base import Model

# Create your models here.
class Contacts(models.Model):
    Name = models.CharField(max_length=255)
    Email = models.EmailField()
    Des = models.TextField(max_length= 255)
    Date = models.DateField()
    
    def __str__(self):
        return self.Name
    
# Create your models here.
