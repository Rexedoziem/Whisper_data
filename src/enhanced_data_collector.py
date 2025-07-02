"""Enhanced Nigerian Pidgin Data Collection Module
with Whisper Automatic Transcription and Persistent Storage
"""

import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
import yt_dlp as youtube_dl
from pathlib import Path
import logging
import argparse
import time
from urllib.parse import urlparse, parse_qs
import re
import random
import whisper
import torch
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPidginDataCollector:
    def __init__(self, output_dir: str = "data/raw", use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'audio').mkdir(exist_ok=True)
        (self.output_dir / 'transcripts').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        # Load Whisper model
        self.whisper_device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        logger.info(f"Loading Whisper model (device: {self.whisper_device})")
        self.whisper_model = whisper.load_model("medium", device=self.whisper_device)
        
        # YouTube-dl configuration
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / 'audio' / '%(id)s.%(ext)s'),  # Use ID for filename
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US'],
            'ignoreerrors': True,
            'http_headers': {
                'User-Agent': random.choice([
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
                ])
            },
        }
    
    def load_collection_progress(self) -> Dict[str, Dict]:
        """Load existing collection progress with video ID as key"""
        progress_file = self.output_dir / 'collection_progress.json'
        if not progress_file.exists():
            return {}
            
        with open(progress_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # Convert list to dict with video ID as key
                return {item['id']: item for item in data}
            except:
                return {}

    def save_collection_progress(self, data: Dict[str, Dict]):
        """Save collection progress as list with video ID as key"""
        progress_file = self.output_dir / 'collection_progress.json'
        # Convert dict to list
        data_list = list(data.values())
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        try:
            query = urlparse(url).query
            params = parse_qs(query)
            return params.get('v', [url.split('/')[-1]])[0].split('&')[0]
        except:
            return url.split('/')[-1].split('?')[0]
    
    def extract_video_info(self, url: str) -> Dict:
        """Extract video information without downloading"""
        video_id = self.extract_video_id(url)
        logger.info(f"Extracting info for video ID: {video_id}")
        
        try:
            with youtube_dl.YoutubeDL({
                'quiet': True,
                'socket_timeout': 30,
                'retries': 5
            }) as ydl:
                info = ydl.extract_info(video_id, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', ''),
                    'view_count': info.get('view_count', 0),
                    'subtitles': info.get('subtitles', {}),
                    'automatic_captions': info.get('automatic_captions', {}),
                    'url': url,
                    'id': video_id
                }
        except youtube_dl.utils.DownloadError as e:
            if "429" in str(e):
                wait_time = 60 * 60  # Wait 1 hour
                logger.error(f"Rate limited. Waiting {wait_time/60} minutes")
                time.sleep(wait_time)
                return self.extract_video_info(url)  # Retry after waiting
            logger.error(f"Download error for {url}: {e}")
            return {'error': str(e), 'url': url, 'id': video_id}
        except Exception as e:
            logger.error(f"Error extracting info from {url}: {e}")
            return {'error': str(e), 'url': url, 'id': video_id}
    
    def has_existing_subtitles(self, video_info: Dict) -> Tuple[bool, Optional[str]]:
        """Check if video has existing subtitles"""
        if 'error' in video_info:
            return False, None
            
        subtitles = video_info.get('subtitles', {})
        auto_captions = video_info.get('automatic_captions', {})
        
        # Check for manual subtitles first (higher quality)
        for lang in ['en', 'en-US', 'en-GB']:
            if lang in subtitles:
                return True, 'manual'
        
        # Check for automatic captions
        for lang in ['en', 'en-US', 'en-GB']:
            if lang in auto_captions:
                return True, 'automatic'
        
        return False, None
    
    def download_media(self, url: str, video_info: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Download audio and subtitles if available"""
        audio_path = None
        subtitle_path = None
        video_id = video_info.get('id', '')
        
        try:
            # Download audio
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([video_id])
            
            # Find downloaded audio file - use ID instead of title
            audio_files = list((self.output_dir / 'audio').glob(f"{video_id}.*"))
            for file in audio_files:
                if file.suffix == '.wav':
                    audio_path = str(file)
                    logger.info(f"Found audio file: {audio_path}")
                    break
            else:
                logger.warning(f"No audio file found for video ID: {video_id}")
            
            # Download subtitles
            subtitle_opts = self.ydl_opts.copy()
            subtitle_opts['skip_download'] = True
            subtitle_opts['writesubtitles'] = True
            subtitle_opts['writeautomaticsub'] = True
            
            with youtube_dl.YoutubeDL(subtitle_opts) as ydl:
                ydl.download([video_id])
                
                # Find downloaded subtitle file
                subtitle_files = list((self.output_dir / 'audio').glob(f"{video_id}.*.vtt"))
                if subtitle_files:
                    subtitle_path = str(subtitle_files[0])
                    logger.info(f"Found subtitle file: {subtitle_path}")
                else:
                    logger.warning(f"No subtitle file found for video ID: {video_id}")
                    
        except youtube_dl.utils.DownloadError as e:
            if "429" in str(e):
                wait_time = 60 * 60  # Wait 1 hour
                logger.error(f"Rate limited during download. Waiting {wait_time/60} minutes")
                time.sleep(wait_time)
                return self.download_media(url, video_info)  # Retry after waiting
            logger.error(f"Download error: {e}")
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
        
        return audio_path, subtitle_path
    
    def transcribe_with_whisper(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper"""
        try:
            logger.info(f"Starting Whisper transcription for: {audio_path}")
            start_time = time.time()
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=True
            )
            
            # Process segments
            segments = []
            for segment in result["segments"]:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
            
            # Get full transcript
            full_text = " ".join([s['text'] for s in segments]).strip()
            
            transcription = {
                'text': full_text,
                'segments': segments,
                'language': result["language"],
                'duration': time.time() - start_time
            }
            
            logger.info(f"Transcription completed in {transcription['duration']:.1f} seconds")
            return transcription
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {'error': str(e)}
    
    def parse_vtt_subtitles(self, vtt_file: str) -> List[Dict]:
        """Parse VTT subtitle file"""
        subtitles = []
        
        try:
            logger.info(f"Parsing VTT file: {vtt_file}")
            with open(vtt_file, 'r', encoding='utf-8-sig') as f:  # Handle BOM
                content = f.read()
            
            # Simple VTT parsing - handle various formats
            blocks = re.split(r'\n\n+', content)
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) < 2:
                    continue
                
                # Skip WEBVTT header
                if lines[0].startswith('WEBVTT'):
                    continue
                
                # Find timestamp line
                timestamp_index = -1
                for i, line in enumerate(lines):
                    if '-->' in line:
                        timestamp_index = i
                        break
                
                if timestamp_index == -1:
                    continue
                
                timestamp = lines[timestamp_index]
                text = ' '.join(lines[timestamp_index+1:])
                
                # Parse timestamps
                start_time, end_time = timestamp.split(' --> ')
                
                # Clean timestamps from positioning info
                start_time = start_time.strip().split(' ', 1)[0]
                end_time = end_time.strip().split(' ', 1)[0]
                
                subtitles.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text.strip()
                })
        
        except Exception as e:
            logger.error(f"Error parsing VTT file {vtt_file}: {e}")
        
        logger.info(f"Parsed {len(subtitles)} subtitles from VTT file")
        return subtitles
    
    def collect_youtube_data(self, url: str, transcribe: bool = True) -> Dict:
        """Collect data from a single YouTube URL"""
        video_id = self.extract_video_id(url)
        logger.info(f"Processing video ID: {video_id}")
        
        try:
            # Extract video info
            video_info = self.extract_video_info(url)
            
            # Handle errors
            if 'error' in video_info:
                return {
                    'url': url,
                    'id': video_id,
                    'status': 'error',
                    'error': video_info['error']
                }
            
            # Check for existing subtitles
            has_subs, sub_type = self.has_existing_subtitles(video_info)
            logger.info(f"Subtitles available: {has_subs} ({sub_type})")
            
            data_item = {
                'url': url,
                'id': video_id,
                'title': video_info.get('title', ''),
                'duration': video_info.get('duration', 0),
                'has_subtitles': has_subs,
                'subtitle_type': sub_type,
                'status': 'pending',
                'source': 'youtube'
            }
            
            # Download media
            audio_path, subtitle_path = self.download_media(url, video_info)
            
            if audio_path:
                data_item['audio_path'] = audio_path
                
                if has_subs and subtitle_path:
                    # Parse subtitles
                    subtitles = self.parse_vtt_subtitles(subtitle_path)
                    
                    if subtitles:
                        # Get full transcript from subtitles
                        full_text = " ".join([s['text'] for s in subtitles]).strip()
                        data_item.update({
                            'subtitles': subtitles,
                            'transcript': full_text,
                            'status': 'ready_for_processing'
                        })
                        logger.info(f"Video ready with {len(subtitles)} subtitles")
                    else:
                        data_item['status'] = 'needs_manual_transcription'
                        logger.warning("Subtitles found but not parsed")
                else:
                    # Use Whisper for transcription if requested
                    if transcribe:
                        whisper_result = self.transcribe_with_whisper(audio_path)
                        if 'text' in whisper_result:
                            data_item.update({
                                'transcript': whisper_result['text'],
                                'whisper_segments': whisper_result['segments'],
                                'status': 'ready_for_processing',
                                'transcription_source': 'whisper'
                            })
                            logger.info("Video ready with Whisper transcription")
                        else:
                            data_item['status'] = 'needs_manual_transcription'
                            data_item['transcription_error'] = whisper_result.get('error', 'Unknown error')
                            logger.warning("Whisper transcription failed")
                    else:
                        data_item['status'] = 'needs_manual_transcription'
                        logger.info("Video needs manual transcription")
            else:
                data_item['status'] = 'error'
                data_item['error'] = 'Media download failed'
                logger.error("Audio download failed")
            
            return data_item
                
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return {
                'url': url,
                'id': video_id,
                'status': 'error',
                'error': str(e),
                'source': 'youtube'
            }
    
    def create_manual_transcription_queue(self, collected_data: List[Dict]) -> str:
        """Create a queue of items needing manual transcription"""
        manual_items = [
            item for item in collected_data 
            if item.get('status') == 'needs_manual_transcription'
        ]
        
        queue_file = self.output_dir / 'manual_transcription_queue.json'
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(manual_items, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created manual transcription queue with {len(manual_items)} items")
        return str(queue_file)
    
    def prepare_training_manifest(self, data: List[Dict], include_metadata: bool = True, full_transcript: bool = False) -> str:
        """Create enhanced training manifest"""
        manifest_path = self.output_dir / 'enhanced_training_manifest.jsonl'
        total_entries = 0
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Skip error items
                if item.get('status') == 'error':
                    continue
                    
                # Handle full transcript mode
                if full_transcript and item.get('transcript'):
                    manifest_entry = {
                        'audio_filepath': item.get('audio_path', ''),
                        'text': item['transcript'],
                        'duration': item.get('duration', 0)
                    }
                    
                    if include_metadata:
                        manifest_entry.update({
                            'source': item.get('source', 'youtube'),
                            'video_title': item.get('title', ''),
                            'subtitle_type': item.get('subtitle_type', 'unknown'),
                            'transcription_source': item.get('transcription_source', 'unknown')
                        })
                    
                    f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                    total_entries += 1
                    continue
                
                # Handle different data sources
                if item.get('source') == 'youtube':
                    # Case 1: Video has subtitles
                    if item.get('subtitles'):
                        for subtitle in item['subtitles']:
                            manifest_entry = {
                                'audio_filepath': item.get('audio_path', ''),
                                'text': subtitle['text'],
                                'duration': self.parse_duration(subtitle['end']) - self.parse_duration(subtitle['start']),
                                'start_time': subtitle['start'],
                                'end_time': subtitle['end']
                            }
                            
                            if include_metadata:
                                manifest_entry.update({
                                    'source': 'youtube',
                                    'video_title': item.get('title', ''),
                                    'subtitle_type': item.get('subtitle_type', 'unknown')
                                })
                            
                            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                            total_entries += 1
                    
                    # Case 2: Whisper transcription available
                    elif item.get('whisper_segments'):
                        for segment in item['whisper_segments']:
                            manifest_entry = {
                                'audio_filepath': item.get('audio_path', ''),
                                'text': segment['text'],
                                'duration': segment['end'] - segment['start'],
                                'start_time': segment['start'],
                                'end_time': segment['end']
                            }
                            
                            if include_metadata:
                                manifest_entry.update({
                                    'source': 'youtube',
                                    'video_title': item.get('title', ''),
                                    'subtitle_type': 'whisper'
                                })
                            
                            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                            total_entries += 1
                    
                    # Case 3: Full manual transcript
                    elif item.get('status') == 'ready_for_processing' and item.get('transcript'):
                        manifest_entry = {
                            'audio_filepath': item.get('audio_path', ''),
                            'text': item['transcript'],
                            'duration': item.get('duration', 0)
                        }
                        
                        if include_metadata:
                            manifest_entry.update({
                                'source': 'youtube',
                                'video_title': item.get('title', ''),
                                'subtitle_type': 'manual'
                            })
                        
                        f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
                        total_entries += 1
        
        logger.info(f"Created enhanced training manifest with {total_entries} entries: {manifest_path}")
        return str(manifest_path)
    
    def parse_duration(self, timestamp: str) -> float:
        """Parse VTT timestamp to seconds"""
        try:
            # Handle comma as decimal separator
            timestamp = timestamp.replace(',', '.')
            
            # Format: HH:MM:SS.mmm or MM:SS.mmm
            parts = timestamp.split(':')
            if len(parts) == 3:  # HH:MM:SS.mmm
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS.mmm
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:  # SS.mmm
                return float(parts[0])
        except Exception as e:
            logger.error(f"Error parsing duration {timestamp}: {e}")
            return 0.0
    
    def generate_collection_report(self, collected_data: List[Dict]) -> Dict:
        """Generate a report of the collection process"""
        report = {
            'total_items': len(collected_data),
            'ready_for_processing': len([d for d in collected_data if d.get('status') == 'ready_for_processing']),
            'needs_manual_transcription': len([d for d in collected_data if d.get('status') == 'needs_manual_transcription']),
            'errors': len([d for d in collected_data if d.get('status') == 'error']),
            'total_duration': sum(d.get('duration', 0) for d in collected_data),
            'sources': {}
        }
        
        # Count transcription sources
        transcription_sources = {}
        for item in collected_data:
            source = item.get('source', 'unknown')
            report['sources'][source] = report['sources'].get(source, 0) + 1
            
            if item.get('subtitle_type'):
                t_source = item['subtitle_type']
                transcription_sources[t_source] = transcription_sources.get(t_source, 0) + 1
            elif item.get('transcription_source'):
                t_source = item['transcription_source']
                transcription_sources[t_source] = transcription_sources.get(t_source, 0) + 1
        
        report['transcription_sources'] = transcription_sources
        
        # Save report
        report_file = self.output_dir / 'collection_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Collection report saved: {report_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Enhanced Nigerian Pidgin Data Collector with Whisper Transcription")
    
    parser.add_argument("--urls-file", help="File containing YouTube URLs (one per line)")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--url", help="Single YouTube URL to process")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Rebuild training manifest from existing data")
    parser.add_argument("--no-transcribe", action="store_true", help="Disable Whisper auto-transcription")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage for Whisper")
    parser.add_argument("--full-transcript", action="store_true", help="Output full transcripts instead of segments")
    
    args = parser.parse_args()
    
    collector = EnhancedPidginDataCollector(
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    # Load existing progress
    progress_data = collector.load_collection_progress()
    
    if args.rebuild_manifest:
        # Convert progress dict to list
        collected_data = list(progress_data.values())
        # Rebuild manifest
        manifest_path = collector.prepare_training_manifest(
            collected_data, 
            full_transcript=args.full_transcript
        )
        print(f"Manifest rebuilt: {manifest_path}")
    
    elif args.url:
        # Process single URL
        video_id = collector.extract_video_id(args.url)
        
        # Only process if not already in progress
        if video_id not in progress_data or progress_data.get(video_id, {}).get('status') == 'error':
            data_item = collector.collect_youtube_data(args.url, not args.no_transcribe)
            progress_data[video_id] = data_item
            collector.save_collection_progress(progress_data)
        else:
            print(f"Skipping {args.url} - already processed")
            data_item = progress_data.get(video_id, {})
        
        # Convert to list for other functions
        collected_data = [data_item]
        
        manifest_path = collector.prepare_training_manifest(
            collected_data,
            full_transcript=args.full_transcript
        )
        queue_path = collector.create_manual_transcription_queue(collected_data)
        report = collector.generate_collection_report(collected_data)
        
        print(f"Data collected. Manifest: {manifest_path}")
        print(f"Manual transcription queue: {queue_path}")
        print(f"Report: {report}")
    
    elif args.urls_file and Path(args.urls_file).exists():
        # Process URLs from file
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        new_items = []
        for url in urls:
            video_id = collector.extract_video_id(url)
            
            # Skip if already processed and not in error state
            if video_id in progress_data and progress_data.get(video_id, {}).get('status') != 'error':
                print(f"Skipping {url} - already processed")
                continue
                
            data = collector.collect_youtube_data(url, not args.no_transcribe)
            progress_data[video_id] = data
            new_items.append(data)
            
            # Save progress after each URL
            collector.save_collection_progress(progress_data)
            
            # Random delay to avoid rate limiting
            time.sleep(random.uniform(5, 15))
        
        # Convert entire progress to list
        collected_data = list(progress_data.values())
        
        manifest_path = collector.prepare_training_manifest(
            collected_data,
            full_transcript=args.full_transcript
        )
        queue_path = collector.create_manual_transcription_queue(collected_data)
        report = collector.generate_collection_report(collected_data)
        
        print(f"Added {len(new_items)} new items")
        print(f"Manifest: {manifest_path}")
        print(f"Manual transcription queue: {queue_path}")
        print(f"Report: {report}")
    
    else:
        print("Please specify a valid command:")
        parser.print_help()

if __name__ == "__main__":
    main()
