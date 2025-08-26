'use client';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useSidebarManager } from '@/app/_common/hooks/useSidebarManager';
import {
    FiSend,
    FiPlus,
    FiFolder,
    FiImage,
    FiMic,
    FiBookmark,
    FiX,
} from 'react-icons/fi';
import styles from '@/app/chat/assets/ChatInterface.module.scss';
import { getWorkflowIOLogs, executeWorkflowById, executeWorkflowByIdStream, loadWorkflow } from '@/app/api/workflowAPI';
import { loadWorkflow as loadWorkflowDeploy } from '@/app/api/workflow/workflowDeployAPI';
import { MessageRenderer } from '@/app/_common/components/ChatParser';
import toast from 'react-hot-toast';
import CollectionModal from '@/app/chat/components/CollectionModal';
import { IOLog, ChatInterfaceProps } from './types';
import ChatHeader from './ChatHeader';
import { ChatArea } from './ChatArea';
import { DeploymentModal } from './DeploymentModal';
import ChatToolsDisplay from './ChatToolsDisplay';
import { generateInteractionId, normalizeWorkflowName } from '@/app/api/interactionAPI';
import { devLog } from '@/app/_common/utils/logger';
import { isStreamingWorkflowFromWorkflow } from '@/app/_common/utils/isStreamingWorkflow';
import { WorkflowData } from '@/app/canvas/types';
import { executeWorkflowByIdDeploy, executeWorkflowByIdStreamDeploy } from '@/app/api/workflow/workflowDeployAPI';
import { SourceInfo } from '../types/source';
import dynamic from 'next/dynamic';
import ResizablePanel from './ResizablePanel';

// Dynamic import to prevent SSR issues with PDF components
const SidePanelPDFViewer = dynamic(() => import('./PDFViewer/SidePanelPDFViewer'), {
  ssr: false,
  loading: () => <div style={{ padding: '20px', textAlign: 'center' }}>PDF 뷰어를 로드하는 중...</div>
});

interface NewChatInterfaceProps extends ChatInterfaceProps {
    onStartNewChat?: (message: string) => void;
    initialMessageToExecute?: string | null;
    user_id?: number | string;
}

const ChatInterface: React.FC<NewChatInterfaceProps> = (
    {
        mode,
        workflow,
        onBack,
        hideBackButton = false,
        existingChatData,
        onStartNewChat,
        initialMessageToExecute,
        user_id,
    }) => {
    const router = useRouter();

    const [ioLogs, setIOLogs] = useState<IOLog[]>([]);
    const [loading, setLoading] = useState(false);
    const [executing, setExecuting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [inputMessage, setInputMessage] = useState<string>('');
    const [pendingLogId, setPendingLogId] = useState<string | null>(null);
    const [showAttachmentMenu, setShowAttachmentMenu] = useState(false);
    const [showCollectionModal, setShowCollectionModal] = useState(false);
    const [selectedCollection, setSelectedCollection] = useState<string[]>([]);
    const [selectedCollectionMakeName, setSelectedCollectionMakeName] = useState<string | null>(null);
    const [collectionMapping, setCollectionMapping] = useState<{ [key: string]: string }>({});
    const [showDeploymentModal, setShowDeploymentModal] = useState(false);
    const [workflowContentDetail, setWorkflowContentDetail] = useState<WorkflowData | null>(null);
    const [additionalParams, setAdditionalParams] = useState<Record<string, Record<string, any>>>({});
    const [showPDFViewer, setShowPDFViewer] = useState(false);
    const [currentSourceInfo, setCurrentSourceInfo] = useState<SourceInfo | null>(null);
    const [panelSplit, setPanelSplit] = useState(() => {
        // localStorage에서 저장된 패널 크기 불러오기
        if (typeof window !== 'undefined') {
            const saved = localStorage.getItem('chatPanelSplit');
            return saved ? parseFloat(saved) : 65;
        }
        return 65;
    });
    const [scrollPosition, setScrollPosition] = useState(0);
    const [isUserScrolling, setIsUserScrolling] = useState(false);
    const [isResizing, setIsResizing] = useState(false);
    const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // showPDFViewer가 false일 때 패널 크기를 100%로 설정
    useEffect(() => {
        if (!showPDFViewer) {
            setPanelSplit(100);
            localStorage.setItem('chatPanelSplit', '100');
        }
    }, [showPDFViewer]);

    const hasExecutedInitialMessage = useRef(false);

    const messagesRef = useRef<HTMLDivElement>(null);
    const attachmentButtonRef = useRef<HTMLDivElement>(null);

    useSidebarManager(showDeploymentModal || showCollectionModal);

    const handleViewSource = (sourceInfo: SourceInfo, messageContent?: string) => {
        // 스크롤 위치 저장 (사용자가 스크롤 중이 아닐 때만)
        if (!isUserScrolling) {
            saveScrollPosition();
        }
        
        // 답변 내용을 sourceInfo에 추가
        const enrichedSourceInfo = {
            ...sourceInfo,
            response_content: messageContent || sourceInfo.response_content
        };
        
        setCurrentSourceInfo(enrichedSourceInfo);
        setShowPDFViewer(true);
        // 출처 뷰어가 켜질 때 패널 크기를 적절히 조정 (65% 채팅, 35% 출처)
        setPanelSplit(65);
        localStorage.setItem('chatPanelSplit', '65');
        
        // 스크롤 위치 복원 (DOM 렌더링 후, 사용자가 스크롤 중이 아닐 때만)
        setTimeout(() => {
            if (!isUserScrolling) {
                restoreScrollPosition();
            }
        }, 150);
    };

    // additionalParams에서 유효한 값만 필터링하는 함수
    const getValidAdditionalParams = useCallback(() => {
        const validParams: Record<string, Record<string, any>> = {};

        Object.keys(additionalParams).forEach(toolId => {
            const toolParams = additionalParams[toolId];
            if (toolParams && typeof toolParams === 'object') {
                const validToolParams: Record<string, any> = {};

                Object.keys(toolParams).forEach(paramKey => {
                    const paramValue = toolParams[paramKey];
                    // null이나 빈 문자열이 아닌 값만 포함
                    if (paramValue !== null && paramValue !== '' && paramValue !== undefined) {
                        validToolParams[paramKey] = paramValue;
                    }
                });

                // 해당 툴에 유효한 파라미터가 있으면 추가
                if (Object.keys(validToolParams).length > 0) {
                    validParams[toolId] = validToolParams;
                }
            }
        });

        return Object.keys(validParams).length > 0 ? validParams : null;
    }, [additionalParams]);

    // 사용자 스크롤 감지 함수
    const handleUserScroll = useCallback(() => {
        if (!isResizing) {
            setIsUserScrolling(true);
            
            // 스크롤 정지 감지
            if (scrollTimeoutRef.current) {
                clearTimeout(scrollTimeoutRef.current);
            }
            
            scrollTimeoutRef.current = setTimeout(() => {
                setIsUserScrolling(false);
                // 스크롤이 정지된 후 현재 위치를 저장
                if (messagesRef.current) {
                    setScrollPosition(messagesRef.current.scrollTop);
                }
            }, 150);
        }
    }, [isResizing]);

    // 스크롤 위치 저장 함수 (조건부)
    const saveScrollPosition = useCallback(() => {
        if (messagesRef.current && !isUserScrolling) {
            setScrollPosition(messagesRef.current.scrollTop);
        }
    }, [isUserScrolling]);

    // 스크롤 위치 복원 함수 (조건부)
    const restoreScrollPosition = useCallback(() => {
        if (messagesRef.current && scrollPosition > 0 && !isUserScrolling && !isResizing) {
            requestAnimationFrame(() => {
                if (messagesRef.current && !isUserScrolling) {
                    messagesRef.current.scrollTop = scrollPosition;
                }
            });
        }
    }, [scrollPosition, isUserScrolling, isResizing]);

    // 강화된 스크롤 함수
    const scrollToBottom = useCallback(() => {
        if (messagesRef.current) {
            // 강제로 스크롤을 최하단으로 이동
            const scrollElement = messagesRef.current;

            // 즉시 스크롤
            scrollElement.scrollTop = scrollElement.scrollHeight;

            // 약간의 지연 후 다시 스크롤 (DOM 업데이트 대기)
            requestAnimationFrame(() => {
                scrollElement.scrollTop = scrollElement.scrollHeight;
            });

            // 추가 보장을 위한 setTimeout
            setTimeout(() => {
                scrollElement.scrollTop = scrollElement.scrollHeight;
            }, 50);
        }
    }, []);

    // workflow 데이터 로드
    useEffect(() => {
        if (workflow && workflow.id && workflow.id !== "default_mode") {
            const loadWorkflowContent = async () => {
                if (user_id) {
                    try {
                        const workflowData = await loadWorkflowDeploy(workflow.name, user_id);
                        setWorkflowContentDetail(workflowData);
                        localStorage.setItem('workflowContentDetail', JSON.stringify(workflowData));
                        devLog.log('Successfully loaded workflow content detail:', workflowData);
                    } catch (error) {
                        devLog.error('Failed to load workflow content detail:', error);
                    }
                } else {
                    try {
                        const workflowData = await loadWorkflow(workflow.name);
                        setWorkflowContentDetail(workflowData);
                        localStorage.setItem('workflowContentDetail', JSON.stringify(workflowData));
                        devLog.log('Successfully loaded workflow content detail:', workflowData);
                    } catch (error) {
                        devLog.error('Failed to load workflow content detail:', error);
                    }
                }
            };

            setLoading(true);
            loadWorkflowContent();
            setLoading(false);
        }
    }, [workflow, user_id]);

    const executeWorkflow = useCallback(async (messageOverride?: string) => {
        console.log('executeWorkflow called')
        if (executing) {
            toast.loading('이전 작업이 완료될 때까지 잠시만 기다려주세요.');
            return;
        }

        const currentMessage = messageOverride || inputMessage;
        if (!currentMessage.trim()) return;

        setInputMessage('');

        setExecuting(true);
        setError(null);
        const tempId = `pending-${Date.now()}`;
        setPendingLogId(tempId);

        setIOLogs((prev) => [
            ...prev,
            {
                log_id: tempId,
                workflow_name: workflow.name,
                workflow_id: workflow.id,
                input_data: currentMessage,
                output_data: '',
                updated_at: new Date().toISOString(),
            },
        ]);

        try {
            let isStreaming: boolean;
            if (workflow.name == 'default_mode') {
                isStreaming = true;
            } else {
                let workflowData = null;

                // 1차: localStorage에서 workflowContentDetail 가져오기 시도
                try {
                    const storedWorkflowData = localStorage.getItem('workflowContentDetail');
                    if (storedWorkflowData) {
                        workflowData = JSON.parse(storedWorkflowData);
                    }
                } catch (error) {
                    console.warn('Failed to load workflow data from localStorage:', error);
                }

                // 2차: state의 workflowContentDetail 사용
                if (!workflowData && workflowContentDetail) {
                    workflowData = workflowContentDetail;
                }

                if (!workflowData) {
                    throw new Error("워크플로우 데이터가 로드되지 않았습니다.");
                }

                isStreaming = await isStreamingWorkflowFromWorkflow(workflowData);
            }

            const { interactionId, workflowId, workflowName } = existingChatData || {
                interactionId: generateInteractionId(),
                workflowId: workflow.id, workflowName: workflow.name
            };

            if (!interactionId || !workflowId || !workflowName) {
                throw new Error("채팅 세션 정보가 유효하지 않습니다.");
            }

            if (isStreaming) {
                await executeWorkflowByIdStream({
                    workflowName,
                    workflowId,
                    inputData: currentMessage,
                    interactionId,
                    selectedCollections: selectedCollection,
                    additional_params: getValidAdditionalParams(),
                    onData: (chunk: string) => {
                        setIOLogs((prev) =>
                            prev.map((log) =>
                                String(log.log_id) === tempId
                                    ? { ...log, output_data: (log.output_data || '') + chunk }
                                    : log
                            )
                        );
                        scrollToBottom();
                    },
                    onEnd: () => setPendingLogId(null),
                    onError: (err: Error) => { throw err; },
                } as any);
            } else {
                const result: any = await executeWorkflowById(workflowName, workflowId, currentMessage, interactionId, selectedCollection, getValidAdditionalParams() as any);
                setIOLogs((prev) =>
                    prev.map((log) =>
                        String(log.log_id) === tempId
                            ? { ...log, output_data: result.outputs ? JSON.stringify(result.outputs[0], null, 2) : result.message || '처리 완료' }
                            : log
                    )
                );
                setPendingLogId(null);
            }
        } catch (err: any) {
            setIOLogs((prev) =>
                prev.map((log) =>
                    String(log.log_id) === tempId
                        ? { ...log, output_data: err.message || '메시지 처리 중 오류 발생' }
                        : log
                )
            );
            setPendingLogId(null);
            toast.error(err.message || '메시지 처리 중 오류가 발생했습니다.');
        } finally {
            setExecuting(false);
            scrollToBottom();
        }
    }, [executing, inputMessage, workflow, existingChatData, scrollToBottom]);

    const executeWorkflowDeploy = useCallback(async (messageOverride?: string) => {
        console.log('executeWorkflowDeploy called')
        if (executing) {
            toast.loading('이전 작업이 완료될 때까지 잠시만 기다려주세요.');
            return;
        }

        const currentMessage = messageOverride || inputMessage;
        if (!currentMessage.trim()) return;

        setInputMessage('');

        setExecuting(true);
        setError(null);
        const tempId = `pending-${Date.now()}`;
        setPendingLogId(tempId);

        setIOLogs((prev) => [
            ...prev,
            {
                log_id: tempId,
                workflow_name: workflow.name,
                workflow_id: workflow.id,
                input_data: currentMessage,
                output_data: '',
                updated_at: new Date().toISOString(),
            },
        ]);

        try {
            let isStreaming: boolean;
            if (workflow.name == 'default_mode') {
                isStreaming = true;
            } else {
                let workflowData = null;

                // 1차: localStorage에서 workflowContentDetail 가져오기 시도
                try {
                    const storedWorkflowData = localStorage.getItem('workflowContentDetail');
                    if (storedWorkflowData) {
                        workflowData = JSON.parse(storedWorkflowData);
                    }
                } catch (error) {
                    console.warn('Failed to load workflow data from localStorage:', error);
                }

                // 2차: state의 workflowContentDetail 사용
                if (!workflowData && workflowContentDetail) {
                    workflowData = workflowContentDetail;
                }

                if (!workflowData) {
                    throw new Error("워크플로우 데이터가 로드되지 않았습니다.");
                }

                isStreaming = await isStreamingWorkflowFromWorkflow(workflowData);
            }

            const { interactionId, workflowId, workflowName } = existingChatData || {
                interactionId: generateInteractionId(),
                workflowId: workflow.id, workflowName: workflow.name
            };

            if (!interactionId || !workflowId || !workflowName) {
                throw new Error("채팅 세션 정보가 유효하지 않습니다.");
            }

            if (isStreaming) {
                await executeWorkflowByIdStreamDeploy({
                    workflowName,
                    workflowId,
                    inputData: currentMessage,
                    interactionId,
                    selectedCollections: selectedCollection,
                    user_id: user_id,
                    additional_params: getValidAdditionalParams(),
                    onData: (chunk) => {
                        setIOLogs((prev) =>
                            prev.map((log) =>
                                String(log.log_id) === tempId
                                    ? { ...log, output_data: (log.output_data || '') + chunk }
                                    : log
                            )
                        );
                        scrollToBottom();
                    },
                    onEnd: () => setPendingLogId(null),
                    onError: (err) => { throw err; },
                });
            } else {
                const result: any = await executeWorkflowByIdDeploy(workflowName, workflowId, currentMessage, interactionId, selectedCollection, user_id, getValidAdditionalParams());
                setIOLogs((prev) =>
                    prev.map((log) =>
                        String(log.log_id) === tempId
                            ? { ...log, output_data: result.outputs ? JSON.stringify(result.outputs[0], null, 2) : result.message || '처리 완료' }
                            : log
                    )
                );
                setPendingLogId(null);
            }
        } catch (err: any) {
            setIOLogs((prev) =>
                prev.map((log) =>
                    String(log.log_id) === tempId
                        ? { ...log, output_data: err.message || '메시지 처리 중 오류 발생' }
                        : log
                )
            );
            setPendingLogId(null);
            toast.error(err.message || '메시지 처리 중 오류가 발생했습니다.');
        } finally {
            setExecuting(false);
            scrollToBottom();
        }
    }, [executing, inputMessage, workflow, existingChatData, scrollToBottom]);

    useEffect(() => {
        if (mode === 'existing' && existingChatData?.interactionId && !initialMessageToExecute) {
            setLoading(true);
            getWorkflowIOLogs(existingChatData.workflowName, existingChatData.workflowId, existingChatData.interactionId)
                .then(logs => {
                    setIOLogs((logs as any).in_out_logs || []);
                })
                .catch(err => {
                    setError('채팅 기록을 불러오는데 실패했습니다.');
                    setIOLogs([]);
                })
                .finally(() => {
                    setLoading(false);

                    executeWorkflow();
                });
        }
    }, [mode, existingChatData]);

    useEffect(() => {
        if (initialMessageToExecute && !hasExecutedInitialMessage.current) {

            hasExecutedInitialMessage.current = true;

            setInputMessage(initialMessageToExecute);

            const newSearchParams = new URLSearchParams(window.location.search);
            newSearchParams.delete('initial_message');
            router.replace(`${window.location.pathname}?${newSearchParams.toString()}`, { scroll: false });
        }
    }, [initialMessageToExecute, executeWorkflow, router]);

    // 스크롤 이벤트 리스너 등록
    useEffect(() => {
        const messagesElement = messagesRef.current;
        if (messagesElement) {
            messagesElement.addEventListener('scroll', handleUserScroll, { passive: true });
            
            return () => {
                messagesElement.removeEventListener('scroll', handleUserScroll);
            };
        }
    }, [handleUserScroll]);

    // 컴포넌트 언마운트 시 타이머 정리
    useEffect(() => {
        return () => {
            if (scrollTimeoutRef.current) {
                clearTimeout(scrollTimeoutRef.current);
            }
        };
    }, []);




    useEffect(() => {
        scrollToBottom();
    }, [ioLogs]);

    // ioLogs 변경 시 강제 스크롤 (스트리밍 중에도 작동)
    useEffect(() => {
        const timer = setTimeout(() => {
            scrollToBottom();
        }, 100);
        return () => clearTimeout(timer);
    }, [ioLogs, executing]);

    // 실행 중일 때 주기적으로 스크롤 (스트리밍 대응)
    useEffect(() => {
        let scrollInterval: NodeJS.Timeout;

        if (executing) {
            // 실행 중일 때 0.5초마다 스크롤 체크
            scrollInterval = setInterval(() => {
                scrollToBottom();
            }, 500);
        }

        return () => {
            if (scrollInterval) {
                clearInterval(scrollInterval);
            }
        };
    }, [executing, scrollToBottom]);

    useEffect(() => {
        if (!showCollectionModal) {
            const checkSelectedCollection = () => {
                try {
                    // 먼저 다중 선택 데이터를 확인
                    const storedMultipleCollections = localStorage.getItem('selectedCollections');
                    if (storedMultipleCollections) {
                        const multipleData = JSON.parse(storedMultipleCollections);
                        if (multipleData.isMultiple && Array.isArray(multipleData.collections)) {
                            setSelectedCollection(multipleData.collections);
                            setCollectionMapping(multipleData.mapping || {});
                            setSelectedCollectionMakeName(multipleData.collections[0]); // 첫 번째를 대표로 표시

                            devLog.log('Loaded multiple collections from localStorage:', {
                                collections: multipleData.collections,
                                mapping: multipleData.mapping,
                                count: multipleData.collections.length
                            });
                            return;
                        }
                    }

                    // 기존 단일 선택 데이터 확인 (호환성)
                    const storedCollection = localStorage.getItem('selectedCollection');
                    if (storedCollection) {
                        const collectionData = JSON.parse(storedCollection);
                        setSelectedCollection([collectionData.name]);
                        setCollectionMapping({ [collectionData.name]: collectionData.make_name });
                        setSelectedCollectionMakeName(collectionData.make_name);

                        devLog.log('Loaded single collection from localStorage (legacy):', {
                            collection: collectionData.name,
                            makeName: collectionData.make_name
                        });
                    } else {
                        setSelectedCollection([]);
                        setCollectionMapping({});
                        setSelectedCollectionMakeName(null);

                        devLog.log('No collections found in localStorage');
                    }
                } catch (err) {
                    console.error('Failed to load selected collection:', err);
                    setSelectedCollection([]);
                    setCollectionMapping({});
                    setSelectedCollectionMakeName(null);
                }
            };
            checkSelectedCollection();
        }
    }, [showCollectionModal]);

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleString('ko-KR', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    /**
     * 메시지 콘텐츠를 렌더링하는 헬퍼 함수
     */
    const renderMessageContent = (content: string, isUserMessage: boolean = false) => {
        if (!content) return null;

        // 답변 내용을 전달하는 헬퍼 함수
        const handleViewSourceWithContext = (sourceInfo: SourceInfo) => {
            handleViewSource(sourceInfo, content);
        };

        return (
            <MessageRenderer
                content={content}
                isUserMessage={isUserMessage}
                onViewSource={handleViewSourceWithContext}
            />
        );
    };

    const handleStartNewChatFlow = useCallback(() => {
        if (!inputMessage.trim() || !onStartNewChat) return;
        const messageToSend = inputMessage;
        setInputMessage('');
        onStartNewChat(messageToSend);
    }, [inputMessage, onStartNewChat]);

    const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey && !executing) {
            e.preventDefault();
            if (mode === 'new-default' || mode === 'new-workflow') {
                handleStartNewChatFlow();
            }
            else if (mode === 'deploy') {
                executeWorkflowDeploy();
            }
            else {
                executeWorkflow();
            }
        }
    }, [executing, mode, handleStartNewChatFlow, executeWorkflow]);

    const handleAttachmentClick = () => {
        setShowAttachmentMenu(!showAttachmentMenu);
    };

    const handleAttachmentOption = (option: string) => {
        console.log('Selected option:', option);
        setShowAttachmentMenu(false);

        if (option === 'collection') {
            setShowCollectionModal(true);
        }
        // TODO: 다른 옵션들에 대한 구현
    };

    // 첨부 메뉴 외부 클릭 시 닫기
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (attachmentButtonRef.current && !attachmentButtonRef.current.contains(event.target as Node)) {
                setShowAttachmentMenu(false);
            }
        };

        if (showAttachmentMenu) {
            document.addEventListener('mousedown', handleClickOutside);
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [showAttachmentMenu]);

    return (
        <div className={styles.container}>
            <ChatHeader
                mode={mode}
                workflow={workflow}
                ioLogs={ioLogs}
                onBack={onBack}
                hideBackButton={hideBackButton}
                onDeploy={() => setShowDeploymentModal(true)}
            />

            {/* Chat Tools Display */}
            <ChatToolsDisplay
                workflowContentDetail={workflowContentDetail}
                additionalParams={additionalParams}
                onAdditionalParamsChange={setAdditionalParams}
            />

            {/* Chat Area with Resizable Panel */}
            <div className={styles.chatContainer}>
                {showPDFViewer && currentSourceInfo ? (
                    <ResizablePanel
                        defaultSplit={panelSplit}
                        minSize={30}
                        maxSize={80}
                        direction="horizontal"
                        onResize={(size) => {
                            // 리사이즈 시작 시 플래그 설정
                            setIsResizing(true);
                            setPanelSplit(size);
                            localStorage.setItem('chatPanelSplit', size.toString());
                            
                            // 리사이즈 완료 후 플래그 해제
                            setTimeout(() => {
                                setIsResizing(false);
                            }, 100);
                        }}
                    >
                        <div className={styles.chatContent}>
                            {/* Chat Area */}
                            <ChatArea
                                mode={mode}
                                loading={loading}
                                ioLogs={ioLogs}
                                workflow={workflow}
                                executing={executing}
                                setInputMessage={setInputMessage}
                                messagesRef={messagesRef}
                                pendingLogId={pendingLogId}
                                renderMessageContent={renderMessageContent}
                                formatDate={formatDate}
                            />

                            <>
                                {/* Collections Display Area - 입력창 위에 위치 */}
                                {selectedCollection.length > 0 && (
                                    <div className={styles.collectionsDisplayArea}>
                                        <div className={styles.collectionsLabel}>
                                            <FiBookmark className={styles.labelIcon} />
                                            <span>선택된 컬렉션</span>
                                        </div>
                                        <div className={styles.selectedCollections}>
                                            {selectedCollection.map((collection, index) => (
                                                <div key={index} className={styles.selectedCollection}>
                                                    <FiBookmark className={styles.collectionIcon} />
                                                    <span className={styles.collectionName}>
                                                        {collectionMapping[collection] || collection}
                                                    </span>
                                                    <button
                                                        className={styles.removeCollectionButton}
                                                        onClick={() => {
                                                            const removedCollection = selectedCollection[index];
                                                            const updatedCollections = selectedCollection.filter((_, i) => i !== index);

                                                            devLog.log('Removing individual collection:', {
                                                                removedCollection,
                                                                previousCollections: selectedCollection,
                                                                updatedCollections,
                                                                remainingCount: updatedCollections.length
                                                            });

                                                            setSelectedCollection(updatedCollections);

                                                            // 매핑에서도 제거된 컬렉션 정보 삭제
                                                            const updatedMapping = { ...collectionMapping };
                                                            delete updatedMapping[removedCollection];
                                                            setCollectionMapping(updatedMapping);

                                                            if (updatedCollections.length === 0) {
                                                                // 모든 컬렉션이 제거된 경우
                                                                localStorage.removeItem('selectedCollections');
                                                                localStorage.removeItem('selectedCollection');
                                                                setSelectedCollectionMakeName(null);
                                                                devLog.log('All collections removed from localStorage');
                                                            } else {
                                                                // 일부 컬렉션이 남아있는 경우
                                                                const multipleCollectionsData = {
                                                                    collections: updatedCollections,
                                                                    mapping: updatedMapping,
                                                                    selectedAt: new Date().toISOString(),
                                                                    isMultiple: true
                                                                };
                                                                localStorage.setItem('selectedCollections', JSON.stringify(multipleCollectionsData));

                                                                // 첫 번째 컬렉션을 단일 선택 형태로도 저장 (호환성)
                                                                const firstCollection = updatedCollections[0];
                                                                const firstMakeName = updatedMapping[firstCollection] || firstCollection;
                                                                localStorage.setItem('selectedCollection', JSON.stringify({
                                                                    name: firstCollection,
                                                                    make_name: firstMakeName,
                                                                    selectedAt: new Date().toISOString()
                                                                }));
                                                                setSelectedCollectionMakeName(firstMakeName);

                                                                devLog.log('Updated collections in localStorage:', multipleCollectionsData);
                                                            }
                                                        }}
                                                        title="컬렉션 해제"
                                                    >
                                                        <FiX />
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Input Area */}
                                <div className={styles.inputArea} style={{ pointerEvents: loading ? 'none' : 'auto' }}>
                                    <div className={styles.inputContainer}>
                                        <input
                                            type="text"
                                            placeholder="메시지를 입력하세요..."
                                            value={inputMessage}
                                            onChange={(e) => setInputMessage(e.target.value)}
                                            onKeyPress={handleKeyPress}
                                            disabled={executing}
                                            className={styles.messageInput}
                                        />
                                        <div className={styles.buttonGroup}>
                                            <div className={styles.attachmentWrapper} ref={attachmentButtonRef}>
                                                <button
                                                    onClick={handleAttachmentClick}
                                                    className={`${styles.attachmentButton} ${showAttachmentMenu ? styles.active : ''}`}
                                                    disabled={executing}
                                                >
                                                    <FiPlus />
                                                </button>
                                                {showAttachmentMenu && (
                                                    <div className={styles.attachmentMenu}>
                                                        <button
                                                            className={styles.attachmentOption}
                                                            onClick={() => handleAttachmentOption('collection')}
                                                        >
                                                            <FiBookmark />
                                                            <span>컬렉션</span>
                                                        </button>
                                                        <button
                                                            className={`${styles.attachmentOption} ${styles.disabled}`}
                                                            disabled
                                                        >
                                                            <FiFolder />
                                                            <span>파일</span>
                                                        </button>
                                                        <button
                                                            className={`${styles.attachmentOption} ${styles.disabled}`}
                                                            disabled
                                                        >
                                                            <FiImage />
                                                            <span>사진</span>
                                                        </button>
                                                        <button
                                                            className={`${styles.attachmentOption} ${styles.disabled}`}
                                                            disabled
                                                        >
                                                            <FiMic />
                                                            <span>음성</span>
                                                        </button>
                                                    </div>
                                                )}
                                            </div>
                                            <button
                                                onClick={() => {
                                                    if (mode === 'new-default' || mode === 'new-workflow') {
                                                        handleStartNewChatFlow();
                                                    }
                                                    else if (mode === 'deploy') {
                                                        executeWorkflowDeploy();
                                                    }
                                                    else {
                                                        executeWorkflow();
                                                    }
                                                }}
                                                disabled={executing || !inputMessage.trim()}
                                                className={`${styles.sendButton} ${executing || !inputMessage.trim() ? styles.disabled : ''}`}
                                            >
                                                {executing ? <div className={styles.miniSpinner}></div> : <FiSend />}
                                            </button>
                                        </div>
                                    </div>
                                    {executing && (
                                        mode === "new-default" ? (
                                            <p className={styles.executingNote}>
                                                일반 채팅을 실행 중입니다...
                                            </p>
                                        ) : (
                                            <p className={styles.executingNote}>
                                                워크플로우를 실행 중입니다...
                                            </p>
                                        )

                                    )}
                                    {error && (
                                        <p className={styles.errorNote}>{error}</p>
                                    )}
                                </div>
                            </>
                        </div>

                        {/* Side Panel for PDF Viewer */}
                        <div className={styles.sidePanel}>
                            <SidePanelPDFViewer
                                sourceInfo={currentSourceInfo}
                                mode={mode}
                                userId={user_id}
                                onClose={() => {
                                    // 스크롤 위치 저장 (사용자가 스크롤 중이 아닐 때만)
                                    if (!isUserScrolling) {
                                        saveScrollPosition();
                                    }
                                    
                                    setShowPDFViewer(false);
                                    setCurrentSourceInfo(null);
                                    // 출처 뷰어 닫힘 시 패널 크기를 초기화하여 채팅 영역이 꽉 차게 설정
                                    setPanelSplit(100);
                                    localStorage.setItem('chatPanelSplit', '100');
                                    
                                    // 스크롤 위치 복원 (DOM 렌더링 후, 사용자가 스크롤 중이 아닐 때만)
                                    setTimeout(() => {
                                        if (!isUserScrolling) {
                                            restoreScrollPosition();
                                        }
                                    }, 150);
                                }}
                            />
                        </div>
                    </ResizablePanel>
                ) : (
                    <div className={styles.chatContent}>
                        {/* Chat Area */}
                        <ChatArea
                            mode={mode}
                            loading={loading}
                            ioLogs={ioLogs}
                            workflow={workflow}
                            executing={executing}
                            setInputMessage={setInputMessage}
                            messagesRef={messagesRef}
                            pendingLogId={pendingLogId}
                            renderMessageContent={renderMessageContent}
                            formatDate={formatDate}
                        />

                        <>
                            {/* Collections Display Area - 입력창 위에 위치 */}
                            {selectedCollection.length > 0 && (
                                <div className={styles.collectionsDisplayArea}>
                                    <div className={styles.collectionsLabel}>
                                        <FiBookmark className={styles.labelIcon} />
                                        <span>선택된 컬렉션</span>
                                    </div>
                                    <div className={styles.selectedCollections}>
                                        {selectedCollection.map((collection, index) => (
                                            <div key={index} className={styles.selectedCollection}>
                                                <FiBookmark className={styles.collectionIcon} />
                                                <span className={styles.collectionName}>
                                                    {collectionMapping[collection] || collection}
                                                </span>
                                                <button
                                                    className={styles.removeCollectionButton}
                                                    onClick={() => {
                                                        const removedCollection = selectedCollection[index];
                                                        const updatedCollections = selectedCollection.filter((_, i) => i !== index);

                                                        devLog.log('Removing individual collection:', {
                                                            removedCollection,
                                                            previousCollections: selectedCollection,
                                                            updatedCollections,
                                                            remainingCount: updatedCollections.length
                                                        });

                                                        setSelectedCollection(updatedCollections);

                                                        // 매핑에서도 제거된 컬렉션 정보 삭제
                                                        const updatedMapping = { ...collectionMapping };
                                                        delete updatedMapping[removedCollection];
                                                        setCollectionMapping(updatedMapping);

                                                        if (updatedCollections.length === 0) {
                                                            // 모든 컬렉션이 제거된 경우
                                                            localStorage.removeItem('selectedCollections');
                                                            localStorage.removeItem('selectedCollection');
                                                            setSelectedCollectionMakeName(null);
                                                            devLog.log('All collections removed from localStorage');
                                                        } else {
                                                            // 일부 컬렉션이 남아있는 경우
                                                            const multipleCollectionsData = {
                                                                collections: updatedCollections,
                                                                mapping: updatedMapping,
                                                                selectedAt: new Date().toISOString(),
                                                                isMultiple: true
                                                            };
                                                            localStorage.setItem('selectedCollections', JSON.stringify(multipleCollectionsData));

                                                            // 첫 번째 컬렉션을 단일 선택 형태로도 저장 (호환성)
                                                            const firstCollection = updatedCollections[0];
                                                            const firstMakeName = updatedMapping[firstCollection] || firstCollection;
                                                            localStorage.setItem('selectedCollection', JSON.stringify({
                                                                name: firstCollection,
                                                                make_name: firstMakeName,
                                                                selectedAt: new Date().toISOString()
                                                            }));
                                                            setSelectedCollectionMakeName(firstMakeName);

                                                            devLog.log('Updated collections in localStorage:', multipleCollectionsData);
                                                        }
                                                    }}
                                                    title="컬렉션 해제"
                                                >
                                                    <FiX />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Input Area */}
                            <div className={styles.inputArea} style={{ pointerEvents: loading ? 'none' : 'auto' }}>
                                <div className={styles.inputContainer}>
                                    <input
                                        type="text"
                                        placeholder="메시지를 입력하세요..."
                                        value={inputMessage}
                                        onChange={(e) => setInputMessage(e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        disabled={executing}
                                        className={styles.messageInput}
                                    />
                                    <div className={styles.buttonGroup}>
                                        <div className={styles.attachmentWrapper} ref={attachmentButtonRef}>
                                            <button
                                                onClick={handleAttachmentClick}
                                                className={`${styles.attachmentButton} ${showAttachmentMenu ? styles.active : ''}`}
                                                disabled={executing}
                                            >
                                                <FiPlus />
                                            </button>
                                            {showAttachmentMenu && (
                                                <div className={styles.attachmentMenu}>
                                                    <button
                                                        className={styles.attachmentOption}
                                                        onClick={() => handleAttachmentOption('collection')}
                                                    >
                                                        <FiBookmark />
                                                        <span>컬렉션</span>
                                                    </button>
                                                    <button
                                                        className={`${styles.attachmentOption} ${styles.disabled}`}
                                                        disabled
                                                    >
                                                        <FiFolder />
                                                        <span>파일</span>
                                                    </button>
                                                    <button
                                                        className={`${styles.attachmentOption} ${styles.disabled}`}
                                                        disabled
                                                    >
                                                        <FiImage />
                                                        <span>사진</span>
                                                    </button>
                                                    <button
                                                        className={`${styles.attachmentOption} ${styles.disabled}`}
                                                        disabled
                                                    >
                                                        <FiMic />
                                                        <span>음성</span>
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                        <button
                                            onClick={() => {
                                                if (mode === 'new-default' || mode === 'new-workflow') {
                                                    handleStartNewChatFlow();
                                                }
                                                else if (mode === 'deploy') {
                                                    executeWorkflowDeploy();
                                                }
                                                else {
                                                    executeWorkflow();
                                                }
                                            }}
                                            disabled={executing || !inputMessage.trim()}
                                            className={`${styles.sendButton} ${executing || !inputMessage.trim() ? styles.disabled : ''}`}
                                        >
                                            {executing ? <div className={styles.miniSpinner}></div> : <FiSend />}
                                        </button>
                                    </div>
                                </div>
                                {executing && (
                                    mode === "new-default" ? (
                                        <p className={styles.executingNote}>
                                            일반 채팅을 실행 중입니다...
                                        </p>
                                    ) : (
                                        <p className={styles.executingNote}>
                                            워크플로우를 실행 중입니다...
                                        </p>
                                    )

                                )}
                                {error && (
                                    <p className={styles.errorNote}>{error}</p>
                                )}
                            </div>
                        </>
                    </div>
                )}
            </div>
            <DeploymentModal
                isOpen={showDeploymentModal}
                onClose={() => setShowDeploymentModal(false)}
                workflow={workflow}
                workflowDetail={workflowContentDetail}
            />

            {/* Collection Modal */}
            <CollectionModal
                isOpen={showCollectionModal}
                onClose={() => setShowCollectionModal(false)}
                onSelectCollections={(collections, mapping = {}) => {
                    devLog.log('Collections selected from modal:', {
                        selectedCollections: collections,
                        count: collections.length,
                        previousSelections: selectedCollection,
                        mapping
                    });

                    setSelectedCollection(collections);
                    setCollectionMapping(mapping);

                    // 선택된 컬렉션들을 localStorage에 배열로 저장
                    if (collections.length > 0) {
                        // 다중 선택 데이터를 저장 (매핑 정보 포함)
                        const multipleCollectionsData = {
                            collections: collections,
                            mapping: mapping,
                            selectedAt: new Date().toISOString(),
                            isMultiple: true
                        };
                        localStorage.setItem('selectedCollections', JSON.stringify(multipleCollectionsData));

                        // 기존 호환성을 위해 첫 번째 컬렉션을 단일 선택 형태로도 저장
                        const firstCollection = collections[0];
                        const firstMakeName = mapping[firstCollection] || firstCollection;
                        localStorage.setItem('selectedCollection', JSON.stringify({
                            name: firstCollection,
                            make_name: firstMakeName,
                            selectedAt: new Date().toISOString()
                        }));
                        setSelectedCollectionMakeName(firstMakeName);

                        devLog.log('Collections saved to localStorage:', {
                            multipleData: multipleCollectionsData,
                            singleCompatibility: firstCollection
                        });
                    } else {
                        // 선택 해제
                        localStorage.removeItem('selectedCollections');
                        localStorage.removeItem('selectedCollection');
                        setSelectedCollectionMakeName(null);
                        setCollectionMapping({});

                        devLog.log('All collections cleared from localStorage');
                    }
                }}
                selectedCollections={selectedCollection}
            />

        </div>
    );
};

export default ChatInterface;
