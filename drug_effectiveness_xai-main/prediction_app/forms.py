from django import forms

AGE_CHOICES = [
    ('0-2', '0-2 years'),
    ('3-12', '3-12 years'),
    ('13-18', '13-18 years'),
    ('19-24', '19-24 years'),
    ('25-34', '25-34 years'),
    ('35-44', '35-44 years'),
    ('45-54', '45-54 years'),
    ('55-64', '55-64 years'),
    ('75 or over', '75 or over'),
]

SEX_CHOICES = [
    ('Male', 'Male'),
    ('Female', 'Female'),
    ('Unknown', 'Prefer not to say'),
]

class PredictionForm(forms.Form):
    age_group = forms.ChoiceField(choices=AGE_CHOICES, label='Age Group')
    sex = forms.ChoiceField(choices=SEX_CHOICES, label='Sex')
    condition = forms.CharField(max_length=100, label='Medical Condition')
    drugs = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}),
        label='Drugs (comma separated)',
        help_text='Enter drugs separated by commas'
    )
    symptoms = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}),
        label='Symptoms (comma separated)',
        help_text='Enter symptoms separated by commas'
    )
