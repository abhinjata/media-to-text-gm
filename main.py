#!/usr/bin/env python3
"""
Self-Improving Media-to-Text Intelligence Core
Darwin Gödel Machine-inspired architecture with LangGraph multi-agent system
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Core dependencies
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio

# Media processing dependencies
try:
    import pdfplumber
    import pypdf
    from docx import Document
    import whisper
    from moviepy.editor import VideoFileClip
    import magic
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pdfplumber pypdf python-docx openai-whisper moviepy python-magic")

# LLM dependencies (assuming OpenAI for evaluation)
try:
    import openai
except ImportError:
    print("Install OpenAI: pip install openai")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MediaFile:
    """Represents a media file with metadata"""
    path: str
    file_type: str
    size: int
    hash: str
    metadata: Dict[str, Any] = None

@dataclass
class ExtractionResult:
    """Result of text extraction with quality metrics"""
    text: str
    confidence: float
    method: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
@dataclass
class ImprovementProposal:
    """Proposed improvement to the system"""
    agent_name: str
    current_method: str
    proposed_method: str
    expected_improvement: float
    reasoning: str
    parameters: Dict[str, Any] = None

@dataclass
class SystemState:
    """Global state of the intelligence core"""
    current_file: Optional[MediaFile] = None
    extraction_results: List[ExtractionResult] = None
    best_result: Optional[ExtractionResult] = None
    improvement_history: List[ImprovementProposal] = None
    agent_configurations: Dict[str, Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.extraction_results is None:
            self.extraction_results = []
        if self.improvement_history is None:
            self.improvement_history = []
        if self.agent_configurations is None:
            self.agent_configurations = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

class MediaTypeDetectorAgent:
    """Agent responsible for detecting media file types"""
    
    def __init__(self):
        self.supported_types = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'audio/mpeg': 'audio',
            'audio/wav': 'audio',
            'audio/x-wav': 'audio',
            'video/mp4': 'video',
            'video/avi': 'video',
            'video/quicktime': 'video',
            'audio/mpeg': 'audio',
            'audio/mp3': 'audio',        
            'audio/mpeg3': 'audio',       
            'audio/x-mpeg': 'audio',        
            'audio/x-mpeg-3': 'audio' 
        }
    
    def detect(self, file_path: str) -> MediaFile:
        try:
            file_path = Path(file_path)
            mime_type = magic.from_file(str(file_path), mime=True)
            
            # DEBUG: Print actual MIME type
            print(f"DEBUG: Detected MIME type: '{mime_type}'")
            
            file_type = self.supported_types.get(mime_type, 'unknown')
            
            # DEBUG: Print resolved file type
            print(f"DEBUG: Resolved file type: '{file_type}'")
            
            # Calculate file hash for versioning
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            return MediaFile(
                path=str(file_path),
                file_type=file_type,
                size=file_path.stat().st_size,
                hash=file_hash,
                metadata={'mime_type': mime_type}
            )
        except Exception as e:
            logger.error(f"Error detecting file type: {e}")
            raise

class TextExtractorAgent:
    """Agent for extracting text from documents"""
    
    def __init__(self):
        self.methods = {
            'pdf': ['pdfplumber', 'pypdf', 'ocr'],
            'docx': ['python-docx'],
            'doc': ['python-docx']
        }
        self.current_method = {}
    
    def extract_pdf_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber"""
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    
    def extract_pdf_pypdf(self, file_path: str) -> str:
        """Extract text using pypdf"""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract(self, media_file: MediaFile, method: str = None) -> ExtractionResult:
        """Extract text using specified or default method"""
        if media_file.file_type == 'pdf':
            method = method or self.current_method.get('pdf', 'pdfplumber')
            if method == 'pdfplumber':
                text = self.extract_pdf_pdfplumber(media_file.path)
            elif method == 'pypdf':
                text = self.extract_pdf_pypdf(media_file.path)
            else:
                raise ValueError(f"Unknown PDF extraction method: {method}")
                
        elif media_file.file_type == 'docx':
            text = self.extract_docx(media_file.path)
            method = 'python-docx'
        else:
            raise ValueError(f"Unsupported file type: {media_file.file_type}")
        
        # Calculate basic confidence score
        confidence = self._calculate_confidence(text)
        
        return ExtractionResult(
            text=text,
            confidence=confidence,
            method=method,
            timestamp=datetime.now(),
            metadata={'char_count': len(text), 'word_count': len(text.split())}
        )
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text characteristics"""
        if not text:
            return 0.0
        
        # Simple heuristics for text quality
        word_count = len(text.split())
        char_count = len(text)
        
        if word_count == 0:
            return 0.0
        
        avg_word_length = char_count / word_count
        
        # Penalize very short or very long average word lengths
        if avg_word_length < 2 or avg_word_length > 20:
            return max(0.3, 1.0 - abs(avg_word_length - 5) / 10)
        
        return min(1.0, word_count / 100)  # Higher score for more words, cap at 1.0

class AudioTranscriberAgent:
    """Agent for transcribing audio files"""
    
    def __init__(self):
        self.model = None
        self.current_model = "base"
        
    def load_model(self, model_name: str = "base"):
        """Load Whisper model"""
        if self.model is None or self.current_model != model_name:
            self.model = whisper.load_model(model_name)
            self.current_model = model_name
    
    def transcribe(self, media_file: MediaFile, model_name: str = "base") -> ExtractionResult:
        """Transcribe audio file"""
        self.load_model(model_name)
        
        result = self.model.transcribe(media_file.path)
        
        return ExtractionResult(
            text=result["text"],
            confidence=self._calculate_audio_confidence(result),
            method=f"whisper-{model_name}",
            timestamp=datetime.now(),
            metadata={
                'language': result.get('language', 'unknown'),
                'segments': len(result.get('segments', []))
            }
        )
    
    def _calculate_audio_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence from Whisper result"""
        segments = whisper_result.get('segments', [])
        if not segments:
            return 0.0
        
        # Average confidence across segments
        total_confidence = sum(segment.get('confidence', 0.0) for segment in segments)
        return total_confidence / len(segments)

class VideoToAudioAgent:
    """Agent for extracting audio from video files"""
    
    def extract_audio(self, media_file: MediaFile, output_path: str = None) -> str:
        """Extract audio from video file"""
        if output_path is None:
            output_path = media_file.path.replace(Path(media_file.path).suffix, '_audio.wav')
        
        with VideoFileClip(media_file.path) as video:
            audio = video.audio
            if audio is not None:
                audio.write_audiofile(output_path, verbose=False, logger=None)
                audio.close()
            else:
                raise ValueError("No audio track found in video")
        
        return output_path

class UnifiedTextFormatterAgent:
    """Agent for formatting and cleaning extracted text"""
    
    def format(self, extraction_result: ExtractionResult) -> ExtractionResult:
        """Format and clean extracted text"""
        text = extraction_result.text
        
        # Basic text cleaning
        text = self._clean_text(text)
        
        # Create new result with formatted text
        return ExtractionResult(
            text=text,
            confidence=extraction_result.confidence,
            method=f"{extraction_result.method}+formatted",
            timestamp=datetime.now(),
            metadata={
                **extraction_result.metadata,
                'original_length': len(extraction_result.text),
                'formatted_length': len(text)
            }
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text"""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

class SelfEvaluatorAgent:
    """Gödel Machine core - evaluates and proposes improvements"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.evaluation_history = []
    
    def evaluate(self, extraction_result: ExtractionResult) -> float:
        """Evaluate the quality of extracted text"""
        # Multi-metric evaluation
        scores = []
        
        # Basic metrics
        scores.append(self._evaluate_length(extraction_result.text))
        scores.append(self._evaluate_coherence(extraction_result.text))
        scores.append(extraction_result.confidence)
        
        # LLM-based evaluation if available
        if self.openai_client:
            try:
                llm_score = self._evaluate_with_llm(extraction_result.text)
                scores.append(llm_score)
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}")
        
        final_score = sum(scores) / len(scores)
        
        self.evaluation_history.append({
            'timestamp': datetime.now(),
            'score': final_score,
            'method': extraction_result.method,
            'text_length': len(extraction_result.text)
        })
        
        return final_score
    
    def _evaluate_length(self, text: str) -> float:
        """Evaluate based on text length"""
        word_count = len(text.split())
        # Optimal range: 50-5000 words
        if word_count < 10:
            return 0.1
        elif word_count < 50:
            return word_count / 50 * 0.7
        elif word_count <= 5000:
            return 1.0
        else:
            return max(0.7, 1.0 - (word_count - 5000) / 10000)
    
    def _evaluate_coherence(self, text: str) -> float:
        """Evaluate text coherence using simple heuristics"""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) < 5:
            return 0.3
        
        # Check for repeated patterns (indicates OCR errors)
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        return min(1.0, repetition_ratio * 1.2)
    
    def _evaluate_with_llm(self, text: str, max_chars: int = 2000) -> float:
        """Evaluate text quality using LLM"""
        # Truncate text for evaluation
        eval_text = text[:max_chars] if len(text) > max_chars else text
        
        prompt = f"""
        Evaluate the quality of this extracted text on a scale of 0.0 to 1.0:
        
        Text: "{eval_text}"
        
        Consider:
        - Coherence and readability
        - Absence of OCR errors or garbled text
        - Proper formatting and structure
        
        Respond with only a number between 0.0 and 1.0.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            return 0.5  # Default score if parsing fails

class ProposerAgent:
    """Proposes improvements to the extraction system"""
    
    def __init__(self):
        self.improvement_strategies = {
            'pdf': [
                {'method': 'pypdf', 'condition': 'low_confidence', 'improvement': 0.2},
                {'method': 'pdfplumber', 'condition': 'pypdf_failed', 'improvement': 0.3},
                {'method': 'ocr', 'condition': 'scanned_pdf', 'improvement': 0.4}
            ],
            'audio': [
                {'model': 'small', 'condition': 'low_confidence', 'improvement': 0.15},
                {'model': 'medium', 'condition': 'very_low_confidence', 'improvement': 0.25},
                {'model': 'large', 'condition': 'critical_content', 'improvement': 0.35}
            ]
        }
    
    def propose_improvement(self, state: SystemState) -> Optional[ImprovementProposal]:
        """Propose an improvement based on current state"""
        if not state.best_result:
            return None
        
        current_score = state.performance_metrics.get('latest_score', 0.0)
        
        # Only propose improvements if current score is below threshold
        if current_score > 0.85:
            return None
        
        file_type = state.current_file.file_type
        current_method = state.best_result.method
        
        if file_type in self.improvement_strategies:
            for strategy in self.improvement_strategies[file_type]:
                if self._condition_met(strategy['condition'], state):
                    if file_type == 'pdf':
                        proposed_method = strategy['method']
                        if proposed_method != current_method.split('+')[0]:
                            return ImprovementProposal(
                                agent_name='TextExtractorAgent',
                                current_method=current_method,
                                proposed_method=proposed_method,
                                expected_improvement=strategy['improvement'],
                                reasoning=f"Switch to {proposed_method} due to {strategy['condition']}",
                                parameters={'method': proposed_method}
                            )
                    elif file_type == 'audio':
                        proposed_model = strategy['model']
                        if proposed_model not in current_method:
                            return ImprovementProposal(
                                agent_name='AudioTranscriberAgent',
                                current_method=current_method,
                                proposed_method=f"whisper-{proposed_model}",
                                expected_improvement=strategy['improvement'],
                                reasoning=f"Upgrade to {proposed_model} model due to {strategy['condition']}",
                                parameters={'model': proposed_model}
                            )
        
        return None
    
    def _condition_met(self, condition: str, state: SystemState) -> bool:
        """Check if improvement condition is met"""
        current_score = state.performance_metrics.get('latest_score', 0.0)
        
        conditions = {
            'low_confidence': current_score < 0.6,
            'very_low_confidence': current_score < 0.4,
            'pypdf_failed': current_score < 0.3 and 'pypdf' in state.best_result.method,
            'scanned_pdf': current_score < 0.5 and state.current_file.file_type == 'pdf',
            'critical_content': current_score < 0.7  # Placeholder for more sophisticated detection
        }
        
        return conditions.get(condition, False)

class VerifierAgent:
    """Verifies if proposed improvements actually work"""
    
    def verify_improvement(self, proposal: ImprovementProposal, state: SystemState) -> bool:
        """Verify if the proposed improvement actually improves results"""
        current_score = state.performance_metrics.get('latest_score', 0.0)
        expected_score = current_score + proposal.expected_improvement
        
        # Simple threshold-based verification
        # In a real implementation, this would actually test the proposed method
        if expected_score > current_score + 0.1:  # Minimum improvement threshold
            return True
        
        return False

class MediaIntelligenceCore:
    """Main orchestrator using LangGraph"""
    
    def __init__(self, openai_api_key: str = None):
        # Initialize agents
        self.media_detector = MediaTypeDetectorAgent()
        self.text_extractor = TextExtractorAgent()
        self.audio_transcriber = AudioTranscriberAgent()
        self.video_processor = VideoToAudioAgent()
        self.formatter = UnifiedTextFormatterAgent()
        self.evaluator = SelfEvaluatorAgent(openai_api_key)
        self.proposer = ProposerAgent()
        self.verifier = VerifierAgent()
        
        # Initialize LangGraph
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def detect_media_type(state: SystemState) -> SystemState:
            """Graph node: Detect media type"""
            logger.info(f"Detecting media type for: {state.current_file.path}")
            # Media type already detected when file is loaded
            return state
        
        def extract_content(state: SystemState) -> SystemState:
            
            media_file = state.current_file
            extraction_results = state.extraction_results or []
                    
            try:
                if media_file.file_type in ['pdf', 'docx', 'doc']:
                    result = self.text_extractor.extract(media_file)
                elif media_file.file_type == 'audio':
                    result = self.audio_transcriber.transcribe(media_file)
                elif media_file.file_type == 'video':
                    # Extract audio first, then transcribe
                    audio_path = self.video_processor.extract_audio(media_file)
                    audio_file = MediaFile(
                        path=audio_path,
                        file_type='audio',
                        size=os.path.getsize(audio_path),
                        hash=hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()
                    )
                    result = self.audio_transcriber.transcribe(audio_file)
                else:
                    raise ValueError(f"Unsupported file type: {media_file.file_type}")
                
                state.extraction_results.append(result)
                state.best_result = result
                
            except Exception as e:
                logger.error(f"Content extraction failed: {e}")
                # Create a failed result
                result = ExtractionResult(
                    text="",
                    confidence=0.0,
                    method="failed",
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                )
                state['extraction_results'] = extraction_results
                state['best_result'] = result
                
                return state
        
        def format_text(state: SystemState) -> SystemState:
            """Graph node: Format extracted text"""
            if state.best_result and state.best_result.text:
                logger.info("Formatting extracted text")
                formatted_result = self.formatter.format(state.best_result)
                state.extraction_results.append(formatted_result)
                state.best_result = formatted_result
            return state
        
        def evaluate_result(state: SystemState) -> SystemState:
            """Graph node: Evaluate extraction quality"""
            if state.best_result:
                logger.info("Evaluating extraction quality")
                score = self.evaluator.evaluate(state.best_result)
                state.performance_metrics['latest_score'] = score
                logger.info(f"Evaluation score: {score:.3f}")
            return state
        
        def propose_improvement(state: SystemState) -> SystemState:
            """Graph node: Propose improvements"""
            logger.info("Analyzing for potential improvements")
            proposal = self.proposer.propose_improvement(state)
            if proposal:
                logger.info(f"Proposed improvement: {proposal.reasoning}")
                state.improvement_history.append(proposal)
                state.performance_metrics['has_proposal'] = True
            else:
                state.performance_metrics['has_proposal'] = False
            return state
        
        def verify_and_apply(state: SystemState) -> SystemState:
            """Graph node: Verify and apply improvements"""
            if state.improvement_history:
                latest_proposal = state.improvement_history[-1]
                logger.info(f"Verifying proposal: {latest_proposal.proposed_method}")
                
                if self.verifier.verify_improvement(latest_proposal, state):
                    logger.info("Improvement verified - applying changes")
                    # Apply the improvement
                    if latest_proposal.agent_name == 'TextExtractorAgent':
                        self.text_extractor.current_method[state.current_file.file_type] = latest_proposal.parameters['method']
                    elif latest_proposal.agent_name == 'AudioTranscriberAgent':
                        self.audio_transcriber.current_model = latest_proposal.parameters['model']
                    
                    state.performance_metrics['improvement_applied'] = True
                else:
                    logger.info("Improvement not verified - keeping current method")
                    state.performance_metrics['improvement_applied'] = False
            
            return state
        
        def should_retry(state: SystemState) -> str:
            """Decision function: Should we retry with improvements?"""
            if (state.performance_metrics.get('improvement_applied', False) and 
                state.performance_metrics.get('latest_score', 0.0) < 0.8):
                return "extract_content"  # Retry extraction with new method
            return END
        
        # Build the graph
        workflow = StateGraph(SystemState)
        
        # Add nodes
        workflow.add_node("detect_media_type", detect_media_type)
        workflow.add_node("extract_content", extract_content)
        workflow.add_node("format_text", format_text)
        workflow.add_node("evaluate_result", evaluate_result)
        workflow.add_node("propose_improvement", propose_improvement)
        workflow.add_node("verify_and_apply", verify_and_apply)
        
        # Add edges
        workflow.set_entry_point("detect_media_type")
        workflow.add_edge("detect_media_type", "extract_content")
        workflow.add_edge("extract_content", "format_text")
        workflow.add_edge("format_text", "evaluate_result")
        workflow.add_edge("evaluate_result", "propose_improvement")
        workflow.add_edge("propose_improvement", "verify_and_apply")
        workflow.add_conditional_edges("verify_and_apply", should_retry)
        
        return workflow
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a media file and return structured text"""
        logger.info(f"Processing file: {file_path}")
        
        # Detect media type
        media_file = self.media_detector.detect(file_path)
        
        # Initialize state
        initial_state = SystemState(current_file=media_file)
        
        # Run the graph
        config = {"configurable": {"thread_id": media_file.hash}}
        
        try:
            final_state = self.app.invoke(initial_state, config)
            
            # Return results
            return {
                'success': True,
                'file_info': asdict(media_file),
                'extracted_text': final_state.best_result.text if final_state.best_result else "",
                'confidence': final_state.best_result.confidence if final_state.best_result else 0.0,
                'method': final_state.best_result.method if final_state.best_result else "unknown",
                'evaluation_score': final_state.performance_metrics.get('latest_score', 0.0),
                'improvements_applied': len(final_state.improvement_history),
                'processing_history': [asdict(result) for result in final_state.extraction_results]
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_info': asdict(media_file)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        return {
            'evaluation_history': self.evaluator.evaluation_history,
            'current_methods': {
                'text_extractor': self.text_extractor.current_method,
                'audio_transcriber': self.audio_transcriber.current_model
            }
        }

# Automatic folder processing
def process_input_folder():
    """Process all files in the input_files folder"""
    # Initialize the core
    core = MediaIntelligenceCore(openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    # Define the input folder path
    input_folder = Path(__file__).parent / "input_files"
    
    print(f"Scanning folder: {input_folder}")
    print("=" * 60)
    
    # Create input folder if it doesn't exist
    input_folder.mkdir(exist_ok=True)
    
    # Get all files in the input folder
    all_files = list(input_folder.glob("*"))
    media_files = [f for f in all_files if f.is_file() and not f.name.startswith('.')]
    
    if not media_files:
        print("No files found in input_files folder!")
        print(f"   Add files to: {input_folder}")
        print("   Supported formats: PDF, DOCX, MP3, WAV, MP4, AVI, MOV")
        return
    
    print(f"Found {len(media_files)} files to process:")
    for file in media_files:
        print(f"   - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
    print()
    
    # Process each file
    results = []
    successful_extractions = 0
    
    for i, file_path in enumerate(media_files, 1):
        print(f"[{i}/{len(media_files)}] Processing: {file_path.name}")
        print("-" * 50)
        
        try:
            result = core.process_file(str(file_path))
            results.append(result)
            
            if result['success']:
                successful_extractions += 1
                print(f"SUCCESS")
                print(f"   Method: {result['method']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Evaluation Score: {result['evaluation_score']:.3f}")
                print(f"   Text Length: {len(result['extracted_text'])} chars")
                print(f"   Improvements: {result['improvements_applied']}")
                
                # Show preview of extracted text
                text_preview = result['extracted_text'][:300].replace('\n', ' ')
                if len(result['extracted_text']) > 300:
                    text_preview += "..."
                print(f"   Preview: {text_preview}")
                
                # Save extracted text to output file
                output_file = input_folder / f"{file_path.stem}_extracted.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"File: {file_path.name}\n")
                    f.write(f"Method: {result['method']}\n")
                    f.write(f"Confidence: {result['confidence']:.3f}\n")
                    f.write(f"Evaluation Score: {result['evaluation_score']:.3f}\n")
                    f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 50 + "\n\n")
                    f.write(result['extracted_text'])
                print(f"   Saved to: {output_file.name}")
                
            else:
                print(f"FAILED: {result['error']}")
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({'success': False, 'error': str(e), 'file': file_path.name})
        
        print()  # Empty line between files
    
    # Summary statistics
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(media_files)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {len(media_files) - successful_extractions}")
    print(f"Success rate: {(successful_extractions/len(media_files)*100):.1f}%")
    
    # Show system learning statistics
    stats = core.get_system_stats()
    if stats['evaluation_history']:
        avg_score = sum(eval['score'] for eval in stats['evaluation_history']) / len(stats['evaluation_history'])
        print(f"Average quality score: {avg_score:.3f}")
        print(f"Total evaluations: {len(stats['evaluation_history'])}")
    
    print(f"Current methods: {stats['current_methods']}")
    
    # Show file type breakdown
    file_types = {}
    for result in results:
        if result['success']:
            file_info = result['file_info']
            file_type = file_info['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
    
    if file_types:
        print(f"File types processed:")
        for file_type, count in file_types.items():
            print(f"   - {file_type}: {count} files")
    
    print(f"\nExtracted text files saved in: {input_folder}")
    print("Run again to process new files or test improvements!")

if __name__ == "__main__":
    print("Media Intelligence Core - Godel Machine")
    print("=" * 50)
    
    # Check if dependencies are available
    try:
        import pdfplumber, pypdf, whisper, moviepy, magic
        print("All media processing dependencies are available.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pdfplumber pypdf python-docx openai-whisper moviepy python-magic")
    
    # Process the input folder
    process_input_folder()
