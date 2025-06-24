from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from ultralytics import YOLO
import easyocr  # Replace pytesseract with easyocr
import cv2
import os
import json
from .models import ViolationRecord
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.db.models import Count, Q
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth, TruncHour, ExtractHour, ExtractMinute
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import requests, json

# Load YOLOv8 models
violation_model = YOLO("C:\\Users\\wanha\\OneDrive\\Documents\\Y3S2\\FYP\\Model Building\\Traffic Violation\\YOLOv8\\YOLOv8 Result (50epoch)\\YOLOv8s_training2\\weights\\best.pt")
plate_model = YOLO("C:\\Users\\wanha\\OneDrive\\Documents\\Y3S2\\FYP\\Model Building\\License Plate\\YOLOv8\\LicensePlate Yolov8 best.pt")

reader = easyocr.Reader(['en'])  # For English license plates

def detect_image(request):
    if request.method == "POST":
        # 1) Grab user inputs
        location         = request.POST.get("location")
        incident_date    = request.POST.get("incidentDate")   # e.g. "2025-04-24"
        incident_time    = request.POST.get("incidentTime")   # e.g. "14:35"
        location_details = request.POST.get("locationDetails")

        # 2) Prepare lists for rendering
        image_files        = request.FILES.getlist("image")
        input_images       = []
        output_images      = []
        all_violations     = []
        all_non_violations = []
        original_paths     = []

        # 3) Ensure media dirs exist
        os.makedirs(os.path.join("media", "outputs"), exist_ok=True)
        os.makedirs(os.path.join("media", "plates"), exist_ok=True)

        for img_file in image_files:
            # 4) Save the uploaded file
            saved_path      = default_storage.save("uploads/" + img_file.name, img_file)
            full_input_path = os.path.join("media", saved_path)
            original_paths.append(full_input_path)

            # 5) Run your model
            results = violation_model(full_input_path)
            violations, non_violations = [], []
            img = cv2.imread(full_input_path)
            non_violation_classes = {
                "person-seatbelt", "With Helmet", "2-or-less-person-on-2-wheeler"
            }

            for res in results:
                if res.boxes:
                    for box in res.boxes:
                        cls   = int(box.cls[0])
                        conf  = float(box.conf[0])
                        label = f"{violation_model.names[cls]} ({conf:.2f})"
                        if violation_model.names[cls] in non_violation_classes:
                            non_violations.append(label)
                        else:
                            violations.append(label)
                    img = res.plot()

            # 6) Write out the annotated image
            out_path = os.path.join("media", "outputs", img_file.name)
            cv2.imwrite(out_path, img)

            # 7) Accumulate for the template
            input_images.append("/media/" + saved_path)
            output_images.append("/media/outputs/" + img_file.name)
            all_violations.append(violations)
            all_non_violations.append(non_violations)

            if violations:
                labels_to_save = violations
            else:
                labels_to_save = [f"No violation - {label}" for label in non_violations]

            # 8) Save a ViolationRecord with **all** fields
            record = ViolationRecord(
                violation_type       = ", ".join(labels_to_save),
                license_plate_number = "Not detected yet",
                original_image       = saved_path,
                detected_image       = "outputs/" + img_file.name,
                location             = location,
                incident_date        = incident_date,
                incident_time        = incident_time,
                location_details     = location_details
            )
            record.save()

        # 9) Render the result
        return render(request, "detection/result.html", {
            "image_data":       zip(input_images, output_images, all_violations, all_non_violations),
            "original_paths":   json.dumps(original_paths),
            "location":         location,
            "incident_date":    incident_date,
            "incident_time":    incident_time,
            "location_details": location_details,
        })

    return render(request, "detection/upload.html")

@csrf_exempt
def detect_license_plate(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    # 1) Grab the list of full paths you passed from result.html:
    image_paths = json.loads(request.POST.get("image_paths", "[]"))

    license_numbers = []
    plate_images    = []

    for full_path in image_paths:
        # 2) Run YOLO to get plate boxes + crop
        img = cv2.imread(full_path)
        plate_results = plate_model(full_path)

        license_number = None
        plate_url      = None

        for pr in plate_results:
            for box in pr.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                if img is None:
                    continue

                cropped = img[y1:y2, x1:x2]
                # 3) Save the crop under media/plates/...
                plate_filename = f"plate_{os.path.basename(full_path)}"
                rel_plate_path = os.path.join("plates", plate_filename).replace("\\", "/")   # e.g. "plates/foo.jpg"
                abs_plate_path = os.path.join(settings.MEDIA_ROOT, rel_plate_path)
                cv2.imwrite(abs_plate_path, cropped)

                # 4) EasyOCR
                texts = [t for _, t, conf in reader.readtext(cropped) if conf > 0.3]
                license_number = " ".join(texts).strip() if texts else "Unknown"

                # 5) Build the URL your front-end will use:
                plate_url = settings.MEDIA_URL + rel_plate_path
                break
            if license_number:
                break

        # 6) If never saw a box:
        if not license_number:
            license_number = "No license plate detected"

        license_numbers.append(license_number)
        plate_images.append(plate_url)  # may be None

        # 7) Update the DB record
        fname      = os.path.basename(full_path)
        rel_upload = f"uploads/{fname}"                                             
        # Atomically update the license_plate_number (and save plate_image if added that field)
        update_kwargs = {"license_plate_number": license_number}
        # if added a dedicated ImageField for the plate crop:
        # update_kwargs["plate_image"] = rel_plate_path
        ViolationRecord.objects \
            .filter(original_image=rel_upload) \
            .update(**update_kwargs)

    # 8) Return the JSON result.html JS expects:
    return JsonResponse({
        "license_numbers": license_numbers,
        "plate_images":    plate_images,
    })



def view_violations(request):
    # Get filter parameters
    violation_types = request.GET.getlist('violation_type')  # Use getlist to handle multiple values
    date_filter = request.GET.get('incidentDate')
    time_filter = request.GET.get('incidentTime')
    location_filter = request.GET.get('location')

    # Initialize queryset
    qs = ViolationRecord.objects.all()

    # Filter by violation type
    if violation_types and violation_types != ['All']:
        q = Q()
        for t in violation_types:
            if t == 'None':
                q |= Q(violation_type__exact='') | Q(violation_type__isnull=True)
            else:
                q |= Q(violation_type__icontains=t)
        qs = qs.filter(q)
    elif not violation_types:
        violation_types = ['All']  # Default to 'All' if no types selected

    # Filter by incident date
    if date_filter:
        try:
            d = datetime.strptime(date_filter, '%Y-%m-%d').date()
            qs = qs.filter(incident_date=d)
        except ValueError:
            pass

    # Filter by incident time
    if time_filter:
        try:
            t = datetime.strptime(time_filter, '%H:%M').time()
            qs = qs.filter(incident_time=t)
        except ValueError:
            pass

    # Filter by location
    if location_filter:
        qs = qs.filter(location=location_filter)

    # Order by incident date and time (most recent first)
    violations = qs.order_by('-incident_date', '-incident_time')

    # Define available violation types
    violation_type_choices = [
        'All',
        'phone',
        'Without Helmet',
        'With Helmet',
        'more-than-2-person-on-2-wheeler',
        '2-or-less-person-on-2-wheeler',
        'person-seatbelt',
        'person-noseatbelt',
        'None',
    ]

    # Prepare context for template
    context = {
        'violations': violations,
        'violation_types': violation_type_choices,
        'selected_violation_types': violation_types if violation_types != ['All'] else 'All',
        'selected_date': date_filter,
        'selected_time': time_filter,
        'selected_location': location_filter,
    }

    return render(request, 'detection/violations.html', context)


def delete_violation(request, record_id):
    if request.method == "POST":
        try:
            record = ViolationRecord.objects.get(id=record_id)
            record.delete()
            return JsonResponse({"status": "success", "message": "Violation successfully deleted."})
        except ViolationRecord.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Violation not found."}, status=404)
    return JsonResponse({"status": "error", "message": "Invalid request method."}, status=400)

@csrf_exempt
def clear_violations(request):
    if request.method == "POST":
        ViolationRecord.objects.all().delete()
        return JsonResponse({"status": "success", "message": "All violations successfully cleared."})
    return redirect('view_violations')

from django.db.models import Q
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth
from django.shortcuts import render
import json

def statistics(request):
    violation_type = request.GET.get('violation_type', 'All')
    time_period = request.GET.get('time_period', 'Day')
    location = request.GET.get('location', '')

    # Normalize violation type for filtering
    if violation_type == 'All':
        violations = ViolationRecord.objects.all()
    else:
        # Use icontains to match partial violation types (e.g., 'person-noseatbelt', 'Without Helmet')
        violations = ViolationRecord.objects.filter(violation_type__icontains=violation_type)

    # Apply location filter
    if location:
        violations = violations.filter(location=location)

    # Apply time period filter using incident_time
    if time_period == 'Day':
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'
    elif time_period == 'Week':
        violations = violations.annotate(period=TruncWeek('incident_date'))
        date_format = 'week'
    elif time_period == 'Month':
        violations = violations.annotate(period=TruncMonth('incident_date'))
        date_format = '%Y-%m'
    elif time_period == 'Morning':
        violations = violations.filter(
            incident_time__hour__gte=6, incident_time__hour__lt=18
        )
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'
    elif time_period == 'Night':
        violations = violations.filter(
            Q(incident_time__hour__gte=18) | Q(incident_time__hour__lt=6)
        )
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'

    violation_counts = {}
    for violation in violations:
        period = violation.period
        period_str = period.strftime(date_format)

        # Split violation types and clean them (remove confidence scores)
        violation_types = [v.strip().split(' (')[0] for v in violation.violation_type.split(',')]
        if violation_type == 'All':
            count = len(violation_types)
        else:
            # Count matches for the selected violation type (partial match)
            count = sum(1 for v in violation_types if violation_type.lower() in v.lower())

        if period_str in violation_counts:
            violation_counts[period_str] += count
        else:
            violation_counts[period_str] = count

    labels = sorted(violation_counts.keys())
    counts = [violation_counts[label] for label in labels]

    locations = [
        'Johor', 'Kedah', 'Kelantan', 'Kuala Lumpur', 'Labuan', 'Melaka',
        'Negeri Sembilan', 'Pahang', 'Penang', 'Perak', 'Perlis', 'Putrajaya',
        'Sabah', 'Sarawak', 'Selangor', 'Terengganu'
    ]

    return render(request, 'detection/statistics.html', {
        'violation_types': ['All', 'phone', 'Without Helmet', 'With Helmet', 'more-than-2-person-on-2-wheeler',
                           '2-or-less-person-on-2-wheeler', 'person-seatbelt', 'person-noseatbelt'],
        'time_periods': ['Day', 'Week', 'Month', 'Morning', 'Night'],
        'locations': locations,
        'selected_type': violation_type,
        'selected_period': time_period,
        'selected_location': location,
        'labels': json.dumps(labels),
        'counts': json.dumps(counts),
    })

from django.db.models import Q
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth
from django.http import JsonResponse

def statistics(request):
    violation_type = request.GET.get('violation_type', 'All')
    time_period = request.GET.get('time_period', 'Day')
    location = request.GET.get('location', '')

    # Normalize violation type for filtering
    if violation_type == 'All':
        violations = ViolationRecord.objects.all()
    else:
        # Use icontains to match partial violation types (e.g., 'person-noseatbelt', 'Without Helmet')
        violations = ViolationRecord.objects.filter(violation_type__icontains=violation_type)

    # Apply location filter
    if location:
        violations = violations.filter(location=location)

    # Apply time period filter using incident_time
    if time_period == 'Day':
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'
    elif time_period == 'Week':
        violations = violations.annotate(period=TruncWeek('incident_date'))
        date_format = '%Y-%m-%d'  # We'll format this as the first day of the week
    elif time_period == 'Month':
        violations = violations.annotate(period=TruncMonth('incident_date'))
        date_format = '%Y-%m'
    elif time_period == 'Morning':
        violations = violations.filter(
            incident_time__hour__gte=6, incident_time__hour__lt=18
        )
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'
    elif time_period == 'Night':
        violations = violations.filter(
            Q(incident_time__hour__gte=18) | Q(incident_time__hour__lt=6)
        )
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'

    violation_counts = {}
    for violation in violations:
        period = violation.period
        if time_period == 'Week':
            # For weeks, format as "Week X of YYYY"
            period_str = f"Week {period.isocalendar()[1]} of {period.year}"
        else:
            period_str = period.strftime(date_format)

        # Split violation types and clean them (remove confidence scores)
        violation_types = [v.strip().split(' (')[0] for v in violation.violation_type.split(',')]
        if violation_type == 'All':
            count = len(violation_types)
        else:
            # Count matches for the selected violation type (partial match)
            count = sum(1 for v in violation_types if violation_type.lower() in v.lower())

        if period_str in violation_counts:
            violation_counts[period_str] += count
        else:
            violation_counts[period_str] = count

    labels = sorted(violation_counts.keys())
    counts = [violation_counts[label] for label in labels]

    locations = [
        'Johor', 'Kedah', 'Kelantan', 'Kuala Lumpur', 'Labuan', 'Melaka',
        'Negeri Sembilan', 'Pahang', 'Penang', 'Perak', 'Perlis', 'Putrajaya',
        'Sabah', 'Sarawak', 'Selangor', 'Terengganu'
    ]

    return render(request, 'detection/statistics.html', {
        'violation_types': ['All', 'phone', 'Without Helmet', 'With Helmet', 'more-than-2-person-on-2-wheeler',
                           '2-or-less-person-on-2-wheeler', 'person-seatbelt', 'person-noseatbelt'],
        'time_periods': ['Day', 'Week', 'Month', 'Morning', 'Night'],
        'locations': locations,
        'selected_type': violation_type,
        'selected_period': time_period,
        'selected_location': location,
        'labels': json.dumps(labels),
        'counts': json.dumps(counts),
    })

def statistics(request):
    violation_type = request.GET.get('violation_type', 'All')
    time_period = request.GET.get('time_period', 'Day')
    location = request.GET.get('location', '')

    # Normalize violation type for filtering
    if violation_type == 'All':
        violations = ViolationRecord.objects.all()
    else:
        # Use icontains to match partial violation types (e.g., 'person-noseatbelt', 'Without Helmet')
        violations = ViolationRecord.objects.filter(violation_type__icontains=violation_type)

    # Apply location filter
    if location:
        violations = violations.filter(location=location)

    # Apply time period filter using incident_time
    if time_period == 'Day':
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'
    elif time_period == 'Week':
        violations = violations.annotate(period=TruncWeek('incident_date'))
        date_format = '%Y-%m-%d'  # We'll format this as the first day of the week
    elif time_period == 'Month':
        violations = violations.annotate(period=TruncMonth('incident_date'))
        date_format = '%Y-%m'
    elif time_period == 'Morning':
        violations = violations.filter(
            incident_time__hour__gte=6, incident_time__hour__lt=18
        )
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'
    elif time_period == 'Night':
        violations = violations.filter(
            Q(incident_time__hour__gte=18) | Q(incident_time__hour__lt=6)
        )
        violations = violations.annotate(period=TruncDay('incident_date'))
        date_format = '%Y-%m-%d'

    violation_counts = {}
    for violation in violations:
        period = violation.period
        if time_period == 'Week':
            # Get first day of the week (Python's date.isocalendar() week starts on Monday)
            # Calculate which week of the month this is
            week_of_month = (period.day - 1) // 7 + 1
            # Format as "Week X of Month YYYY"
            month_name = period.strftime('%b')  # Abbreviated month name
            period_str = f"Week {week_of_month} of {month_name} {period.year}"
        else:
            period_str = period.strftime(date_format)

        # Split violation types and clean them (remove confidence scores)
        violation_types = [v.strip().split(' (')[0] for v in violation.violation_type.split(',')]
        if violation_type == 'All':
            count = len(violation_types)
        else:
            # Count matches for the selected violation type (partial match)
            count = sum(1 for v in violation_types if violation_type.lower() in v.lower())

        if period_str in violation_counts:
            violation_counts[period_str] += count
        else:
            violation_counts[period_str] = count

    labels = sorted(violation_counts.keys())
    counts = [violation_counts[label] for label in labels]

    locations = [
        'Johor', 'Kedah', 'Kelantan', 'Kuala Lumpur', 'Labuan', 'Melaka',
        'Negeri Sembilan', 'Pahang', 'Penang', 'Perak', 'Perlis', 'Putrajaya',
        'Sabah', 'Sarawak', 'Selangor', 'Terengganu'
    ]

    return render(request, 'detection/statistics.html', {
        'violation_types': ['All', 'phone', 'Without Helmet', 'With Helmet', 'more-than-2-person-on-2-wheeler',
                           '2-or-less-person-on-2-wheeler', 'person-seatbelt', 'person-noseatbelt'],
        'time_periods': ['Day', 'Week', 'Month', 'Morning', 'Night'],
        'locations': locations,
        'selected_type': violation_type,
        'selected_period': time_period,
        'selected_location': location,
        'labels': json.dumps(labels),
        'counts': json.dumps(counts),
    })

def statistics_data(request):
    violation_type = request.GET.get('violation_type', 'All')
    time_period = request.GET.get('time_period', 'Day')
    location = request.GET.get('location', '')

    if time_period == 'Day':
        period_annotate = TruncDay('incident_date')
        date_format = '%Y-%m-%d'
    elif time_period == 'Week':
        period_annotate = TruncWeek('incident_date')
        date_format = '%Y-%m-%d'  # We'll handle this specially
    elif time_period == 'Month':
        period_annotate = TruncMonth('incident_date')
        date_format = '%Y-%m'
    elif time_period == 'Morning' or time_period == 'Night':
        period_annotate = TruncDay('incident_date')
        date_format = '%Y-%m-%d'
    else:
        period_annotate = TruncDay('incident_date')
        date_format = '%Y-%m-%d'

    # Normalize violation type for filtering
    if violation_type == 'All':
        violations = ViolationRecord.objects.all()
    else:
        # Use icontains to match partial violation types (e.g., 'person-noseatbelt', 'Without Helmet')
        violations = ViolationRecord.objects.filter(violation_type__icontains=violation_type)

    if location:
        violations = violations.filter(location=location)

    if time_period == 'Morning':
        violations = violations.filter(
            incident_time__hour__gte=6, incident_time__hour__lt=18
        )
    elif time_period == 'Night':
        violations = violations.filter(
            Q(incident_time__hour__gte=18) | Q(incident_time__hour__lt=6)
        )

    violations = violations.annotate(period=period_annotate)

    violation_counts = {}
    for violation in violations:
        period = violation.period
        if time_period == 'Week':
            # Get first day of the week (Python's date.isocalendar() week starts on Monday)
            # Calculate which week of the month this is
            week_of_month = (period.day - 1) // 7 + 1
            # Format as "Week X of Month YYYY"
            month_name = period.strftime('%b')  # Abbreviated month name
            period_str = f"Week {week_of_month} of {month_name} {period.year}"
        else:
            period_str = period.strftime(date_format)

        # Split violation types and clean them (remove confidence scores)
        violation_types = [v.strip().split(' (')[0] for v in violation.violation_type.split(',')]
        if violation_type == 'All':
            count = len(violation_types)
        else:
            # Count matches for the selected violation type (partial match)
            count = sum(1 for v in violation_types if violation_type.lower() in v.lower())

        if period_str in violation_counts:
            violation_counts[period_str] += count
        else:
            violation_counts[period_str] = count

    labels = sorted(violation_counts.keys())
    values = [violation_counts[label] for label in labels]

    return JsonResponse({'labels': labels, 'values': values})

@csrf_exempt
def get_recommendations(request):
    if request.method == 'POST':
        try:
            # Load JSON body at the start
            data = json.loads(request.body)

            # Extract image and chart data
            image_data = data.get('image')
            chart1 = data.get("chart1", {})
            chart2 = data.get("chart2", {})

            if not image_data:
                return JsonResponse({"error": "No image data provided"}, status=400)

            chart_description = f"""
Chart 1 - {chart1.get('title')}:
Time Period: {chart1.get('timePeriod', 'Unknown')}
Labels: {chart1.get('labels')}
Values: {chart1.get('values')}

Chart 2 - {chart2.get('title')}:
Time Period: {chart2.get('timePeriod', 'Unknown')}
Labels: {chart2.get('labels')}
Values: {chart2.get('values')}
"""

            api_key = settings.OPENROUTER_API_KEY
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Referer": f"https://{request.META.get('HTTP_HOST', 'localhost')}",
            }

            payload = {
                "model": "qwen/qwen2.5-vl-72b-instruct:free",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a highly advanced traffic data analyst. The user will provide you with a screenshot "
                            "containing two comparison graphs of traffic violations. Your job is to thoroughly analyze "
                            "the trends, differences, and correlations in the image and provide detailed insights and "
                            "recommendations for improving traffic safety based on the graphs."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Please analyze the two traffic violation charts shown in the image and provide a comprehensive analysis with the following sections:\n\n"
                                    "1. KEY TRENDS: Identify the most significant patterns or trends in each chart.\n"
                                    "2. COMPARATIVE ANALYSIS: Compare and contrast the data between the two charts. Highlight any correlations, differences, or relationships.\n"
                                    "3. HOTSPOTS & TIMING: Identify locations, time periods, or conditions with notably high violation rates.\n"
                                    "4. ACTIONABLE RECOMMENDATIONS: Provide specific, practical recommendations for reducing violations and improving traffic safety based on the data. Include both immediate actions and long-term strategies.\n\n"
                                    f"Chart data details:\n{chart_description}\n\n"
                                    "Format your recommendations as clear, concise bullet points organized under headings. Focus on data-driven insights that traffic enforcement officials could implement."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                return JsonResponse({"error": f"API Error: {response.text}"}, status=500)

            result = response.json()
            recommendations = result["choices"][0]["message"]["content"]

            return JsonResponse({"recommendations": recommendations})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)