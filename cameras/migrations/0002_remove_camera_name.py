# Generated by Django 4.0.2 on 2022-04-11 14:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cameras', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='camera',
            name='name',
        ),
    ]
