# Generated by Django 3.1 on 2021-05-22 21:10

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('model_training', '0005_auto_20210519_0329'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='modeltraining',
            options={'ordering': ['-time'], 'verbose_name': 'model training', 'verbose_name_plural': 'model trainings'},
        ),
    ]
