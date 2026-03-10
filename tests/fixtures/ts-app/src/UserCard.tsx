import React from 'react'
import type { User } from './types'

interface UserCardProps {
    user: User
    onSelect?: (user: User) => void
}

export const UserCard = ({ user, onSelect }: UserCardProps) => {
    return (
        <div className="user-card" onClick={() => onSelect?.(user)}>
            <h3>{user.name}</h3>
            <p>{user.email}</p>
        </div>
    )
}

export function UserList({ users }: { users: User[] }) {
    return (
        <div className="user-list">
            {users.map(u => (
                <UserCard key={u.id} user={u} />
            ))}
        </div>
    )
}

const UserBadge = ({ user }: { user: User }) => {
    return <span className="badge">{user.name[0]}</span>
}

export const MemoizedCard = React.memo(function MemoCard({ user }: { user: User }) {
    return <div>{user.name}</div>
})

export const ForwardedInput = React.forwardRef<HTMLInputElement, { label: string }>(
    ({ label }, ref) => (
        <label>
            {label}
            <input ref={ref} />
        </label>
    )
)

export default UserCard
