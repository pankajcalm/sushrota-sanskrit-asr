#!/usr/bin/env python3
"""Pre-warm the Vāgbodhinī sample ślokas into the TTS cache so first users get instant playback."""
import requests
S = ["वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\nदेवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥",
     "या कुन्देन्दुतुषारहारधवला या शुभ्रवस्त्रावृता ।\nया वीणावरदण्डमण्डितकरा या श्वेतपद्मासना ॥",
     "हठलुठ दल घिष्टोत्कण्ठदष्टोष्ठ विद्युत्\nसटशठ कठिनोरः पीठभित्सुष्ठुनिष्ठाम् ।\nपठतिनुतव कण्ठाधिष्ठ घोरान्त्रमाला\nदह दह नरसिंहासह्यवीर्याहितं मे ॥",
     "ಗುರುರ್ಬ್ರಹ್ಮಾ ಗುರುರ್ವಿಷ್ಣುಃ ಗುರುರ್ದೇವೋ ಮಹೇಶ್ವರಃ ।\nಗುರುಃ ಸಾಕ್ಷಾತ್ ಪರಂ ಬ್ರಹ್ಮ ತಸ್ಮೈ ಶ್ರೀಗುರವೇ ನಮಃ ॥",
     "సరస్వతి నమస్తుభ్యం వరదే కామరూపిణి ।\nవిద్యారమ్భం కరిష్యామి సిద్ధిర్భవతు మే సదా ॥"]
for i, t in enumerate(S):
    r = requests.post("http://localhost:8010/generate", data={"text": t}, stream=True)
    done = None
    for line in r.iter_lines():
        if line and b'"done"' in line: done = True
    print(f"sample {i+1}: {'ok' if done else 'FAIL'}")
print("PREWARM DONE")
