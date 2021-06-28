# Generated by Django 3.1 on 2021-05-19 07:52

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('model_training', '0005_auto_20210519_0329'),
    ]

    operations = [
        migrations.CreateModel(
            name='Metric',
            fields=[
                ('created', models.DateTimeField(editable=False, null=True)),
                ('updated', models.DateTimeField(editable=False, null=True)),
                ('id', models.CharField(editable=False, max_length=64, primary_key=True, serialize=False)),
                ('name', models.CharField(editable=False, max_length=256)),
                ('value', models.FloatField()),
                ('training', models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='metric_relation', to='model_training.modeltraining')),
            ],
            options={
                'verbose_name': 'metric',
                'verbose_name_plural': 'metrics',
                'db_table': 'model_manager_metric',
                'ordering': ['name'],
                'abstract': False,
                'unique_together': {('training', 'name')},
            },
        ),
    ]