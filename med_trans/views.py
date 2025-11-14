import os

from openai import OpenAI
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class GetTreatmentView(APIView):
    def post(self, request):
        description = request.data.get("description")
        if not description:
            return Response(
                {"error": "Missing 'description' in request body."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        prompt = f"""
        Extract patient age and recommend treatment plan based on this medical transcription:
        {description}
        Format the response as JSON with keys 'age' and 'treatment'.
        """

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            ai_response = completion.choices[0].message.content

            return Response({"response": ai_response}, status=status.HTTP_200_OK)

        except Exception:
            return Response(
                {"error": str(Exception)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ICDCodeView(APIView):
    def post(self, request):
        treatment = request.data.get("treatment")
        if not treatment:
            return Response(
                {"error": "Missing 'treatment' in request body."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        prompt = f"""
        Given the following treatment description:
        {treatment}

        Identify the most relevant ICD-10 code.
        Return only JSON with key 'icd_code' and the code as value.
        """

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            ai_response = completion.choices[0].message.content

            return Response({"response": ai_response}, status=status.HTTP_200_OK)

        except Exception:
            return Response(
                {"error": str(Exception)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
