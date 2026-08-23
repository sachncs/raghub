/**
 * Conversation turn — one message in a session-scoped history.
 *
 * Sessions are namespaced by `userId::rawSessionToken`. Two callers
 * who share or guess a `SessionId` cannot read each other's history
 * because the store path filters on the joined `userId`.
 */

import type { SessionId, WorkspaceId, UserId } from './ids.js';

export const TurnRole = {
  User: 'user',
  Assistant: 'assistant',
  System: 'system',
  Tool: 'tool',
} as const;

export type TurnRoleValue = (typeof TurnRole)[keyof typeof TurnRole];

export interface TurnProps {
  readonly sessionId: SessionId;
  readonly workspaceId: WorkspaceId;
  readonly userId: UserId;
  readonly role: TurnRoleValue;
  readonly content: string;
  readonly createdAt: Date;
}

export class Turn {
  private readonly props: TurnProps;

  constructor(props: TurnProps) {
    this.props = Object.freeze({ ...props });
  }

  public get sessionId(): SessionId {
    return this.props.sessionId;
  }

  public get workspaceId(): WorkspaceId {
    return this.props.workspaceId;
  }

  public get userId(): UserId {
    return this.props.userId;
  }

  public get role(): TurnRoleValue {
    return this.props.role;
  }

  public get content(): string {
    return this.props.content;
  }

  public get createdAt(): Date {
    return this.props.createdAt;
  }

  public toJSON(): TurnProps {
    return { ...this.props, createdAt: this.props.createdAt };
  }
}