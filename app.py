import os
import json
import uuid
import time
import re
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from fpdf import FPDF
import io

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# In-memory storage for serverless compatibility
meetings = {}

class MeetingStore:
    def __init__(self, meeting_type='physical'):
        self.meeting_type = meeting_type
        self.participants = {}
        self.speaker_counter = 0
        self.transcript = []
        self.unanswered_questions = []
        self.start_time = None
        self.end_time = None
        self.meeting_title = f"Meeting - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.audio_sources = set()
        
    def get_or_create_speaker(self, voice_features, audio_source='default'):
        """Advanced speaker recognition with source isolation"""
        source_key = f"{audio_source}_{voice_features.get('channel', 'mono')}"
        
        best_match = None
        best_score = float('inf')
        
        search_space = list(self.participants.items())
        
        for speaker_id, data in search_space:
            pitch_diff = abs(data.get('avg_pitch', 0) - voice_features.get('avg_pitch', 0))
            pace_diff = abs(data.get('avg_pace', 0) - voice_features.get('words_per_minute', 0)) / 10
            energy_diff = abs(data.get('avg_energy', 0) - voice_features.get('energy', 0)) / 50
            
            score = (pitch_diff * 0.5) + (pace_diff * 0.3) + (energy_diff * 0.2)
            
            threshold = 25 if self.meeting_type == 'hybrid' else 20
            if score < threshold and score < best_score:
                best_score = score
                best_match = speaker_id
                
        if best_match:
            old_count = len([t for t in self.transcript if t['speaker_id'] == best_match])
            speaker_data = self.participants[best_match]
            
            alpha = 0.3
            speaker_data['avg_pitch'] = (alpha * voice_features.get('avg_pitch', 100) + 
                                          (1-alpha) * speaker_data.get('avg_pitch', 100))
            speaker_data['avg_pace'] = (alpha * voice_features.get('words_per_minute', 120) + 
                                       (1-alpha) * speaker_data.get('avg_pace', 120))
            speaker_data['samples_count'] = old_count + 1
            return best_match
            
        self.speaker_counter += 1
        speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
        
        is_remote = audio_source in ['tab_audio', 'system_audio', 'screen_share']
        
        self.participants[speaker_id] = {
            "name": f"Participant {self.speaker_counter}" + (" (Remote)" if is_remote else ""),
            "avg_pitch": voice_features.get('avg_pitch', 100),
            "avg_pace": voice_features.get('words_per_minute', 120),
            "avg_energy": voice_features.get('energy', 5000),
            "voice_id": speaker_id,
            "audio_source": audio_source,
            "is_remote": is_remote,
            "samples_count": 1,
            "first_seen": datetime.now().isoformat()
        }
        return speaker_id
    
    def is_question(self, text):
        """Enhanced question detection"""
        text = text.strip()
        text_lower = text.lower()
        
        if text.endswith('?'):
            return True
            
        starters = ['what', 'where', 'when', 'why', 'who', 'how', 'is', 'are', 'can', 
                   'could', 'would', 'should', 'will', 'do', 'does', 'did', 'have', 
                   'has', 'am', 'was', 'were']
        words = text_lower.split()
        if words and words[0] in starters:
            return True
            
        patterns = [
            r'\b(what|where|when|why|how|who)\s+(is|are|was|were|do|does|did|can|could|would)',
            r'\b(can you|could you|would you|will you)\s+\w+',
            r'\b(let me know|tell me|i\'m wondering|i was wondering)\b',
            r'\b(any idea|do you know|have you heard)\b'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
    
    def detect_decision(self, text):
        """Auto-detect decisions"""
        indicators = ['decided', 'decision', 'agreed', 'conclusion', 'finalized', 
                     'resolved', 'approved', 'confirmed', 'consensus', 'settled on',
                     'moving forward with', 'we will', 'let\'s go with', 'approved']
        text_lower = text.lower()
        return any(ind in text_lower for ind in indicators)
    
    def detect_action_item(self, text, speaker):
        """Auto-detect action items"""
        assign_patterns = [
            r'(i will|i\'ll|i am going to|i\'m going to)\s+(.+)',
            r'(let me|i can|i should)\s+(.+)',
            r'(will you|can you|could you)\s+(.+)',
            r'(action item|todo|task|follow up|follow-up)'
        ]
        
        text_lower = text.lower()
        
        for pattern in assign_patterns:
            if re.search(pattern, text_lower):
                return True
                
        deadline_words = ['by tomorrow', 'by friday', 'by monday', 'by next', 
                         'by the end of', 'asap', 'soon', 'this week', 'next week']
        if any(d in text_lower for d in deadline_words):
            return True
            
        return False
    
    def add_utterance(self, text, voice_features, audio_source='default'):
        """Process new speech"""
        speaker_id = self.get_or_create_speaker(voice_features, audio_source)
        speaker_data = self.participants[speaker_id]
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'speaker_id': speaker_id,
            'speaker_name': speaker_data['name'],
            'text': text,
            'type': 'statement',
            'audio_source': audio_source,
            'is_remote': speaker_data['is_remote'],
            'voice_features': {
                'pitch': voice_features.get('avg_pitch'),
                'pace': voice_features.get('words_per_minute'),
                'energy': voice_features.get('energy')
            }
        }
        
        if self.is_question(text):
            entry['type'] = 'question'
            entry['question_id'] = len(self.transcript)
            self.unanswered_questions.append({
                'index': len(self.transcript),
                'timestamp': datetime.now(),
                'speaker': speaker_id
            })
        else:
            if self.unanswered_questions:
                last_q = self.unanswered_questions[-1]
                time_since_q = len(self.transcript) - last_q['index']
                
                is_answer = (
                    time_since_q <= 4 and 
                    last_q['speaker'] != speaker_id and 
                    not text.endswith('?')
                )
                
                answer_starters = ['yes', 'no', 'absolutely', 'definitely', 'correct', 
                                 'right', 'exactly', 'sure', 'well', 'i think', 'in my opinion']
                if any(text.lower().startswith(starter) for starter in answer_starters):
                    is_answer = True
                    
                if is_answer:
                    entry['type'] = 'answer'
                    entry['answers_question'] = last_q['index']
                    
                    if time_since_q >= 2 or text.endswith('.') or len(text) > 100:
                        self.unanswered_questions.pop()
                        
        if self.detect_decision(text):
            entry['is_decision'] = True
            
        if self.detect_action_item(text, speaker_id):
            entry['is_action_item'] = True
            entry['assignee'] = speaker_data['name']
                        
        self.transcript.append(entry)
        self.audio_sources.add(audio_source)
        return entry
    
    def get_minutes_structure(self):
        """Generate structured minutes"""
        minutes = {
            'title': self.meeting_title,
            'date': self.start_time or datetime.now().isoformat(),
            'meeting_type': self.meeting_type,
            'duration': self.calculate_duration(),
            'participants': [],
            'remote_participants': [],
            'qa_pairs': [],
            'key_discussion_points': [],
            'decisions': [],
            'action_items': [],
            'audio_sources': list(self.audio_sources)
        }
        
        for pid, pdata in self.participants.items():
            participant_info = {
                'name': pdata['name'],
                'source': pdata['audio_source'],
                'speaking_time': len([t for t in self.transcript if t['speaker_id'] == pid])
            }
            if pdata['is_remote']:
                minutes['remote_participants'].append(participant_info)
            else:
                minutes['participants'].append(participant_info)
        
        used_indices = set()
        for i, entry in enumerate(self.transcript):
            if entry['type'] == 'question':
                qa = {
                    'question': entry,
                    'answers': [],
                    'follow_up_questions': []
                }
                used_indices.add(i)
                
                j = i + 1
                while j < len(self.transcript):
                    next_entry = self.transcript[j]
                    if next_entry.get('answers_question') == i:
                        qa['answers'].append(next_entry)
                        used_indices.add(j)
                        j += 1
                    elif next_entry['type'] == 'question' and j < len(self.transcript) - 1:
                        if next_entry['speaker_id'] == entry['speaker_id']:
                            qa['follow_up_questions'].append(next_entry)
                            used_indices.add(j)
                        break
                    else:
                        break
                        
                minutes['qa_pairs'].append(qa)
        
        for i, entry in enumerate(self.transcript):
            if i in used_indices:
                continue
                
            if entry.get('is_decision'):
                minutes['decisions'].append(entry)
            elif entry.get('is_action_item'):
                minutes['action_items'].append(entry)
            elif entry['type'] == 'statement' and len(entry['text']) > 50:
                minutes['key_discussion_points'].append(entry)
                
        return minutes
    
    def calculate_duration(self):
        if not self.end_time or not self.start_time:
            return "In progress"
        try:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            duration = end - start
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except:
            return "Unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/meeting/start', methods=['POST'])
def start_meeting():
    try:
        data = request.get_json() or {}
        meeting_type = data.get('meeting_type', 'physical')
        
        meeting_id = str(uuid.uuid4())
        meetings[meeting_id] = MeetingStore(meeting_type=meeting_type)
        meetings[meeting_id].start_time = datetime.now().isoformat()
        
        return jsonify({
            'meeting_id': meeting_id,
            'meeting_type': meeting_type,
            'status': 'started',
            'timestamp': meetings[meeting_id].start_time,
            'message': get_setup_message(meeting_type)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_setup_message(meeting_type):
    messages = {
        'physical': 'Place your device centrally in the room. All voices will be captured via microphone.',
        'online': 'Share your meeting tab/window when prompted. System audio will be captured automatically.',
        'hybrid': 'Dual-mode active: Microphone captures room audio while screen share captures remote participants.'
    }
    return messages.get(meeting_type, 'Meeting started.')

@app.route('/api/meeting/<meeting_id>/audio', methods=['POST'])
def receive_audio(meeting_id):
    try:
        if meeting_id not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '').strip()
        voice_features = data.get('voice_features', {
            'avg_pitch': 100,
            'words_per_minute': 120,
            'energy': 5000
        })
        audio_source = data.get('audio_source', 'microphone')
        channel = data.get('channel', 'mono')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        meeting = meetings[meeting_id]
        voice_features['channel'] = channel
        
        entry = meeting.add_utterance(text, voice_features, audio_source)
        
        return jsonify({
            'entry': entry,
            'speaker_count': len(meeting.participants),
            'audio_sources': list(meeting.audio_sources)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/meeting/<meeting_id>/status', methods=['GET'])
def get_status(meeting_id):
    try:
        if meeting_id not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
            
        meeting = meetings[meeting_id]
        recent = request.args.get('since', 0, type=int)
        
        new_entries = meeting.transcript[recent:]
        
        return jsonify({
            'entries': new_entries,
            'total_count': len(meeting.transcript),
            'participants': {
                k: {
                    'name': v['name'], 
                    'is_remote': v['is_remote'],
                    'source': v['audio_source']
                } 
                for k, v in meeting.participants.items()
            },
            'audio_sources': list(meeting.audio_sources),
            'is_active': meeting.end_time is None,
            'meeting_type': meeting.meeting_type,
            'unanswered_questions': len(meeting.unanswered_questions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/meeting/<meeting_id>/stop', methods=['POST'])
def stop_meeting(meeting_id):
    try:
        if meeting_id not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
            
        meetings[meeting_id].end_time = datetime.now().isoformat()
        return jsonify({
            'status': 'stopped',
            'duration': meetings[meeting_id].calculate_duration()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/meeting/<meeting_id>/pdf', methods=['GET'])
def generate_pdf(meeting_id):
    try:
        if meeting_id not in meetings:
            return jsonify({'error': 'Meeting not found'}), 404
            
        meeting = meetings[meeting_id]
        minutes = meeting.get_minutes_structure()
        
        # Create PDF with error handling
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Helper function to safely encode text
        def safe_text(text):
            if text is None:
                return ""
            # Encode to latin-1, replacing unsupported characters
            return text.encode('latin-1', 'replace').decode('latin-1')
        
        # Header
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 12, safe_text('MEETING MINUTES'), ln=True, align='C')
        
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(100, 100, 100)
        date_str = datetime.now().strftime('%B %d, %Y at %H:%M')
        pdf.cell(0, 6, safe_text(f"{date_str} | {minutes['meeting_type'].upper()} MEETING"), ln=True, align='C')
        pdf.line(10, 35, 200, 35)
        pdf.ln(8)
        
        # Meeting Info
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, safe_text('Meeting Information'), ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, safe_text(f"Duration: {minutes['duration']}"), ln=True)
        pdf.cell(0, 6, safe_text(f"Audio Sources: {', '.join(minutes['audio_sources'])}"), ln=True)
        pdf.ln(4)
        
        # Participants Section
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, safe_text(' PARTICIPANTS'), ln=True, fill=True)
        
        if minutes['participants']:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, safe_text('In-Person:'), ln=True)
            pdf.set_font('Arial', '', 10)
            for p in minutes['participants']:
                line = f"  - {p['name']} ({p['speaking_time']} contributions)"
                pdf.cell(0, 5, safe_text(line), ln=True)
        
        if minutes['remote_participants']:
            pdf.ln(2)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, safe_text('Remote:'), ln=True)
            pdf.set_font('Arial', '', 10)
            for p in minutes['remote_participants']:
                source = p['source'].replace('_', ' ').title()
                line = f"  - {p['name']} via {source}"
                pdf.cell(0, 5, safe_text(line), ln=True)
        
        pdf.ln(6)
        
        # Q&A Section
        if minutes['qa_pairs']:
            pdf.set_font('Arial', 'B', 12)
            pdf.set_fill_color(230, 240, 255)
            pdf.cell(0, 10, safe_text(' QUESTIONS & ANSWERS'), ln=True, fill=True)
            pdf.ln(2)
            
            for i, qa in enumerate(minutes['qa_pairs'], 1):
                # Question
                pdf.set_font('Arial', 'B', 10)
                pdf.set_text_color(0, 51, 102)
                q_time = qa['question']['timestamp'][11:16] if len(qa['question']['timestamp']) > 16 else qa['question']['timestamp']
                q_text = qa['question']['text']
                q_speaker = qa['question']['speaker_name']
                
                pdf.multi_cell(0, 6, safe_text(f"Q{i}. [{q_time}] {q_speaker}:"))
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 5, safe_text(f"    {q_text}"))
                
                # Answers
                pdf.set_text_color(0, 102, 51)
                for ans in qa['answers']:
                    a_time = ans['timestamp'][11:16] if len(ans['timestamp']) > 16 else ans['timestamp']
                    a_speaker = ans['speaker_name']
                    a_text = ans['text']
                    pdf.set_font('Arial', 'B', 9)
                    pdf.cell(0, 5, safe_text(f"    A. [{a_time}] {a_speaker}:"), ln=True)
                    pdf.set_font('Arial', '', 9)
                    pdf.multi_cell(0, 4, safe_text(f"       {a_text}"))
                
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)
        
        # Key Decisions
        if minutes['decisions']:
            pdf.ln(4)
            pdf.set_font('Arial', 'B', 12)
            pdf.set_fill_color(255, 240, 230)
            pdf.cell(0, 10, safe_text(' KEY DECISIONS'), ln=True, fill=True)
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(180, 80, 40)
            
            for d in minutes['decisions']:
                time_str = d['timestamp'][11:16] if len(d['timestamp']) > 16 else d['timestamp']
                pdf.multi_cell(0, 6, safe_text(f"• [{time_str}] {d['text']}"))
            pdf.set_text_color(0, 0, 0)
        
        # Action Items
        if minutes['action_items']:
            pdf.ln(4)
            pdf.set_font('Arial', 'B', 12)
            pdf.set_fill_color(255, 255, 230)
            pdf.cell(0, 10, safe_text(' ACTION ITEMS'), ln=True, fill=True)
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(180, 140, 40)
            
            for item in minutes['action_items']:
                time_str = item['timestamp'][11:16] if len(item['timestamp']) > 16 else item['timestamp']
                assignee = item.get('assignee', 'Unassigned')
                pdf.multi_cell(0, 6, safe_text(f"• [{time_str}] {assignee}: {item['text']}"))
            pdf.set_text_color(0, 0, 0)
        
        # Additional Discussion
        other_entries = [e for e in meeting.transcript 
                         if e['type'] == 'statement' and len(e['text']) > 30 
                         and not e.get('is_decision') and not e.get('is_action_item')]
        
        if other_entries:
            pdf.ln(4)
            pdf.set_font('Arial', 'B', 12)
            pdf.set_fill_color(245, 245, 245)
            pdf.cell(0, 10, safe_text(' ADDITIONAL DISCUSSION'), ln=True, fill=True)
            pdf.set_font('Arial', '', 9)
            
            for entry in other_entries[:15]:  # Limit to prevent huge PDFs
                time_str = entry['timestamp'][11:16] if len(entry['timestamp']) > 16 else entry['timestamp']
                text = entry['text'][:80] + '...' if len(entry['text']) > 80 else entry['text']
                line = f"[{time_str}] {entry['speaker_name']}: {text}"
                pdf.cell(0, 5, safe_text(line), ln=True)
        
        # Footer
        pdf.set_y(-15)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, safe_text(f'Generated by VoiceMinutes AI | Page {pdf.page_no()}'), 0, 0, 'C')
        
        # Output to bytes
        output = io.BytesIO()
        pdf.output(output)
        output.seek(0)
        
        # Generate safe filename
        safe_meeting_type = meeting.meeting_type.replace('/', '-')
        filename = f'meeting_minutes_{safe_meeting_type}_{meeting_id[:8]}.pdf'
        
        return send_file(
            output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        import traceback
        print(f"PDF Generation Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))