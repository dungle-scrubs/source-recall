import { Database } from './db'
import type { User, CreateUserRequest } from './types'

export class UserService {
    private db: Database

    constructor(db: Database) {
        this.db = db
    }

    async findById(id: string): Promise<User | null> {
        const row = await this.db.query('SELECT * FROM users WHERE id = ?', [id])
        if (!row) return null
        return this.mapToUser(row)
    }

    async create(request: CreateUserRequest): Promise<User> {
        const id = crypto.randomUUID()
        await this.db.execute(
            'INSERT INTO users (id, name, email) VALUES (?, ?, ?)',
            [id, request.name, request.email]
        )
        return { id, ...request, createdAt: new Date() }
    }

    async validateEmail(email: string): Promise<boolean> {
        const existing = await this.db.query(
            'SELECT id FROM users WHERE email = ?',
            [email]
        )
        return !existing
    }

    private mapToUser(row: Record<string, unknown>): User {
        return {
            id: row.id as string,
            name: row.name as string,
            email: row.email as string,
            createdAt: new Date(row.created_at as string),
        }
    }
}
