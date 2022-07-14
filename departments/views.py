from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View
from .forms import NewClusterForm
from. models import Department
from django.contrib import messages
# Create your views here.

class AddNewClusterView(View):
    def get(self, request):
        form = NewClusterForm()
        context = {}
        deptList = Department.objects.all()
        context['deptList'] = deptList
        context['form'] = form
        return render(request, 'departments/newCluster.html', context)

    def post(self, request):
        form = NewClusterForm(request.POST)
        if form.is_valid():
            form.save()
        else:
            messages.error(request, "Form did not validate")
            return redirect(reverse('new-cluster'))
