import React, { useState, useRef, useEffect } from "react";
import { useParams, useNavigate } from "react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Car, AlertCircle, Loader2, SendHorizontal, ArrowUpRight } from "lucide-react";
import {
    sendChatMessage,
    getChatHistory,
    createChat,
    isAxiosError,
    type Message,
} from "@/api/client";
import { toast } from "sonner";

export default function ChatPage(): React.ReactElement {
    const { chatId: chatId } = useParams<{ chatId: string }>();
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [chatIdInput, setChatIdInput] = useState<string>("");
    const scrollRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const navigate = useNavigate();

    useEffect(() => {
        // don't fetch history if not in a chat or in a chat that already started
        if (!chatId || messages.length > 0) return;

        getChatHistory(chatId)
            .then(setMessages)
            .catch(() => {
                // New session, no history yet
            });
    }, [chatId]);

    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    useEffect(() => {
        if (!isLoading) {
            inputRef.current?.focus();
        }
    }, [isLoading]);

    // figure out if chat id typed by user is valid and navigate if yes
    const goToChatId = async (): Promise<void> => {
        const targetId = chatIdInput.trim();
        try {
            await getChatHistory(targetId);
            navigate(`/chat/${targetId}`);
        } catch {
            toast.error("Chat not found");
        }
    }

    const sendMessage = async (): Promise<void> => {
        if (!input.trim() || isLoading) return;

        const userMessageContent: string = input.trim();

        if (!chatId) {
            const response = await createChat(userMessageContent);
            navigate(`/chat/${response.chat_id}`, {replace: true});
            return;
        }

        setMessages((prev) => [...prev, { role: "user", content: userMessageContent }]);
        setInput("");
        setIsLoading(true);
        setError(null);

        try {
            const response = await sendChatMessage(chatId, input.trim());
            setMessages((prev) => [...prev, response.message]);
        } catch (err) {
            if (isAxiosError(err)) {
                const errorMessage =
                    err.response?.data?.detail ||
                    err.message ||
                    "Something went wrong";
                setError(errorMessage);
            } else {
                setError("An unexpected error occurred");
            }
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const suggestedQuestions: string[] = [
        "What is Lane Keeping Assist?",
        "How does Adaptive Cruise Control work?",
        "Was ist Abstandsregelung?",
        "Are these systems safe to rely on?",
    ];

    return (
        <div className="flex flex-col h-screen overflow-hidden bg-background">
            {/* Header */}
            <div className="border-b bg-primary text-primary-foreground p-4">
                <div className="flex items-center gap-3 max-w-3xl mx-auto">
                    <Car className="h-7 w-7" />
                    <div>
                        <h1 className="text-xl font-semibold">Beifahrer</h1>
                        <p className="text-sm opacity-80">
                            Driver assistance systems guide
                        </p>
                    </div>
                    <div className="ml-auto text-foreground rounded-md">
                        <div className="flex items-center">
                            <Input
                                className="w-25 bg-white flex-1 rounded-r-none border-r-0"
                                placeholder="Chat ID"
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                                    setChatIdInput(e.target.value)
                                }
                                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                                    if (e.key === "Enter" && /^\d{6}$/.test(chatIdInput.trim())) {
                                        e.preventDefault();
                                        goToChatId();
                                    }
                                }}
                                maxLength={6}
                            />
                            <Button
                                variant="outline"
                                className="rounded-l-none"
                                onClick={goToChatId}
                                disabled={!/^\d{6}$/.test(chatIdInput.trim())}
                            >
                                <ArrowUpRight className="h-4 w-4" />
                            </Button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Messages Area */}
            <ScrollArea className="flex-1 min-h-0 px-4 py-0">
                <div className="max-w-3xl mx-auto space-y-4 py-4">
                    {messages.length === 0 && (
                        <Card className="border-dashed">
                            <CardHeader className="text-center pb-2">
                                <Car className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                                <CardTitle>Welcome to Beifahrer</CardTitle>
                            </CardHeader>
                            <CardContent className="text-center">
                                <p className="text-muted-foreground mb-6">
                                    Ask me about lane assist, adaptive cruise
                                    control, collision warning, and other ADAS
                                    features.
                                </p>
                                <div className="flex flex-wrap justify-center gap-2">
                                    {suggestedQuestions.map((q, i) => (
                                        <Button
                                            key={i}
                                            variant="outline"
                                            size="sm"
                                            onClick={() => setInput(q)}
                                        >
                                            {q}
                                        </Button>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {messages.map((msg, i) => (
                        <div
                            key={i}
                            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                            <Card
                                className={`max-w-[80%] py-0 ${
                                    msg.role === "user"
                                        ? "bg-primary text-primary-foreground"
                                        : "bg-muted"
                                }`}
                            >
                                <CardContent className="p-3">
                                    <p className="whitespace-pre-wrap">
                                        {msg.content}
                                    </p>
                                </CardContent>
                            </Card>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex justify-start">
                            <Card className="bg-muted py-0">
                                <CardContent className="p-3">
                                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                                </CardContent>
                            </Card>
                        </div>
                    )}

                    {error && (
                        <Alert variant="destructive">
                            <AlertCircle className="h-4 w-4" />
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}
                    <div ref={scrollRef} />
                </div>
            </ScrollArea>

            {/* Input Area */}
            <div className="border-t bg-background p-4">
                <div className="max-w-3xl mx-auto flex">
                    <Input
                        ref={inputRef}
                        value={input}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setInput(e.target.value)
                        }
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about driver assistance systems..."
                        disabled={isLoading}
                        className="bg-white flex-1 rounded-r-none border-r-0"
                    />
                    <Button
                        onClick={sendMessage}
                        disabled={isLoading || !input.trim()}
                        className="rounded-l-none"
                    >
                        {isLoading ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                            <SendHorizontal className="h-4 w-4" />
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
}
