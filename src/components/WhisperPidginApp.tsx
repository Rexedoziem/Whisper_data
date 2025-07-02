import React, { useState, useRef, useEffect } from 'react';
import { 
  Mic, 
  Upload, 
  Play, 
  Pause, 
  Download, 
  Settings, 
  Database, 
  Brain, 
  FileText,
  Youtube,
  Headphones,
  Save,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';

interface TranscriptionItem {
  id: string;
  audioUrl: string;
  text: string;
  confidence?: number;
  duration: number;
  status: 'pending' | 'transcribed' | 'verified';
  source: 'upload' | 'youtube' | 'recording';
}

interface TrainingProgress {
  phase: string;
  progress: number;
  status: string;
  logs: string[];
}

const WhisperPidginApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'collect' | 'transcribe' | 'train' | 'test'>('collect');
  const [transcriptions, setTranscriptions] = useState<TranscriptionItem[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [modelStatus, setModelStatus] = useState<'untrained' | 'training' | 'ready'>('untrained');
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);

  // Simulated API calls (replace with actual backend calls)
  const extractFromYoutube = async (url: string) => {
    // Simulate YouTube extraction
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          success: true,
          audioUrl: '/sample-audio.wav',
          existingSubtitles: Math.random() > 0.7 ? 'How you dey today?' : null,
          duration: 30
        });
      }, 2000);
    });
  };

  const startTraining = async () => {
    setModelStatus('training');
    setTrainingProgress({
      phase: 'Data Preparation',
      progress: 0,
      status: 'Starting...',
      logs: ['Initializing training pipeline...']
    });

    // Simulate training phases
    const phases = [
      'Data Preparation',
      'Model Loading',
      'Fine-tuning',
      'Evaluation',
      'Model Saving'
    ];

    for (let i = 0; i < phases.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 3000));
      setTrainingProgress(prev => ({
        ...prev!,
        phase: phases[i],
        progress: ((i + 1) / phases.length) * 100,
        status: `Processing ${phases[i]}...`,
        logs: [...prev!.logs, `Completed ${phases[i]}`]
      }));
    }

    setModelStatus('ready');
    setTrainingProgress(null);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks(prev => [...prev, event.data]);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const saveRecording = () => {
    if (recordedChunks.length > 0) {
      const blob = new Blob(recordedChunks, { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      
      const newTranscription: TranscriptionItem = {
        id: Date.now().toString(),
        audioUrl: url,
        text: '',
        duration: 10, // Estimate
        status: 'pending',
        source: 'recording'
      };

      setTranscriptions(prev => [...prev, newTranscription]);
      setRecordedChunks([]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg">
                <Headphones className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Whisper Pidgin Trainer</h1>
                <p className="text-sm text-gray-600">Fine-tune Whisper for Nigerian Pidgin</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                modelStatus === 'ready' ? 'bg-green-100 text-green-800' :
                modelStatus === 'training' ? 'bg-yellow-100 text-yellow-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {modelStatus === 'ready' ? 'Model Ready' :
                 modelStatus === 'training' ? 'Training...' :
                 'Model Not Trained'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'collect', label: 'Data Collection', icon: Database },
              { id: 'transcribe', label: 'Transcription', icon: FileText },
              { id: 'train', label: 'Training', icon: Brain },
              { id: 'test', label: 'Testing', icon: Mic }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-5 w-5" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'collect' && (
          <div className="space-y-8">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <Youtube className="h-6 w-6 text-red-500 mr-3" />
                YouTube Data Collection
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    YouTube URL
                  </label>
                  <div className="flex space-x-3">
                    <input
                      type="url"
                      placeholder="https://youtube.com/watch?v=..."
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
                      Extract
                    </button>
                  </div>
                </div>
                
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                    <div className="text-sm text-blue-800">
                      <p className="font-medium mb-1">How it works:</p>
                      <ul className="list-disc list-inside space-y-1">
                        <li>Extracts audio from YouTube videos</li>
                        <li>Automatically detects existing subtitles if available</li>
                        <li>If no subtitles found, queues for manual transcription</li>
                        <li>Supports batch processing of multiple URLs</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <Upload className="h-6 w-6 text-blue-500 mr-3" />
                Audio File Upload
              </h2>
              
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-900 mb-2">Drop audio files here</p>
                <p className="text-gray-600 mb-4">or click to browse</p>
                <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  Choose Files
                </button>
                <p className="text-xs text-gray-500 mt-2">Supports WAV, MP3, FLAC, M4A</p>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <Mic className="h-6 w-6 text-green-500 mr-3" />
                Record Audio
              </h2>
              
              <div className="flex items-center justify-center space-x-4">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    className="flex items-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <Mic className="h-5 w-5" />
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="flex items-center space-x-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors animate-pulse"
                  >
                    <div className="h-3 w-3 bg-white rounded-full"></div>
                    <span>Stop Recording</span>
                  </button>
                )}
                
                {recordedChunks.length > 0 && (
                  <button
                    onClick={saveRecording}
                    className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <Save className="h-5 w-5" />
                    <span>Save Recording</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'transcribe' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Manual Transcription Tool</h2>
              
              {transcriptions.length === 0 ? (
                <div className="text-center py-12">
                  <FileText className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No audio files to transcribe yet.</p>
                  <p className="text-sm text-gray-400 mt-2">Add some audio files in the Data Collection tab.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {transcriptions.map((item) => (
                    <div key={item.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            item.status === 'verified' ? 'bg-green-500' :
                            item.status === 'transcribed' ? 'bg-yellow-500' :
                            'bg-gray-300'
                          }`}></div>
                          <span className="text-sm font-medium text-gray-700">
                            {item.source === 'youtube' ? 'YouTube' :
                             item.source === 'upload' ? 'Upload' :
                             'Recording'} • {item.duration}s
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <button className="p-2 text-gray-400 hover:text-blue-600 transition-colors">
                            <Play className="h-4 w-4" />
                          </button>
                          <button className="p-2 text-gray-400 hover:text-green-600 transition-colors">
                            <CheckCircle className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                      
                      <textarea
                        placeholder="Type the Nigerian Pidgin transcription here..."
                        value={item.text}
                        onChange={(e) => {
                          setTranscriptions(prev => 
                            prev.map(t => t.id === item.id ? { ...t, text: e.target.value } : t)
                          );
                        }}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                        rows={3}
                      />
                      
                      <div className="mt-3 flex justify-between items-center">
                        <div className="text-xs text-gray-500">
                          Common phrases: "How you dey?", "I dey fine o", "Wetin dey happen?"
                        </div>
                        <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">
                          Mark as Complete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'train' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <Brain className="h-6 w-6 text-purple-500 mr-3" />
                Model Training
              </h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Training Configuration</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Base Model
                      </label>
                      <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                        <option>openai/whisper-small</option>
                        <option>openai/whisper-medium</option>
                        <option>openai/whisper-large</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Training Epochs
                      </label>
                      <input
                        type="number"
                        defaultValue={10}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Batch Size
                      </label>
                      <input
                        type="number"
                        defaultValue={4}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Dataset Statistics</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Audio Files:</span>
                      <span className="font-medium">{transcriptions.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Transcribed:</span>
                      <span className="font-medium">
                        {transcriptions.filter(t => t.status !== 'pending').length}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total Duration:</span>
                      <span className="font-medium">
                        {transcriptions.reduce((sum, t) => sum + t.duration, 0)}s
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-8">
                {trainingProgress ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">
                        {trainingProgress.phase}
                      </span>
                      <span className="text-sm text-gray-500">
                        {Math.round(trainingProgress.progress)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${trainingProgress.progress}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-600">{trainingProgress.status}</p>
                    
                    <div className="bg-gray-50 rounded-lg p-4 max-h-32 overflow-y-auto">
                      {trainingProgress.logs.map((log, index) => (
                        <p key={index} className="text-xs text-gray-600 font-mono">
                          {log}
                        </p>
                      ))}
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={startTraining}
                    disabled={transcriptions.filter(t => t.status !== 'pending').length === 0}
                    className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
                  >
                    <Brain className="h-5 w-5" />
                    <span>Start Training</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'test' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <Mic className="h-6 w-6 text-green-500 mr-3" />
                Test Your Model
              </h2>
              
              {modelStatus !== 'ready' ? (
                <div className="text-center py-12">
                  <AlertCircle className="h-16 w-16 text-yellow-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-2">Model not ready for testing</p>
                  <p className="text-sm text-gray-500">Complete training first to test your model.</p>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                      <Mic className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <p className="font-medium text-gray-900 mb-2">Record & Test</p>
                      <p className="text-sm text-gray-600 mb-4">Speak in Nigerian Pidgin</p>
                      <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                        Start Recording
                      </button>
                    </div>
                    
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                      <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <p className="font-medium text-gray-900 mb-2">Upload & Test</p>
                      <p className="text-sm text-gray-600 mb-4">Test with audio file</p>
                      <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                        Choose File
                      </button>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-6">
                    <h3 className="font-medium text-gray-900 mb-4">Sample Test Phrases</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {[
                        "How you dey?",
                        "I dey fine o",
                        "Wetin you wan chop?",
                        "Make we go market",
                        "Abeg help me",
                        "Na so e be"
                      ].map((phrase, index) => (
                        <div key={index} className="flex items-center justify-between bg-white p-3 rounded-lg">
                          <span className="text-sm">{phrase}</span>
                          <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                            Test
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WhisperPidginApp;