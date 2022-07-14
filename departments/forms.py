from django.forms import ModelForm
from.models import Department

class NewClusterForm(ModelForm):
    class Meta:
        model = Department
        fields = '__all__'
