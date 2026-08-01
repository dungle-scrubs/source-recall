<template>
  <article class="review-card">
    <header class="review-card__header">
      <h3>{{ review.title }}</h3>
      <StarRating :value="review.rating" />
    </header>

    <p class="review-card__body">{{ review.body }}</p>

    <footer v-if="canModerate">
      <button type="button" @click="approve">Approve</button>
      <button type="button" @click="reject">Reject</button>
    </footer>
  </article>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import StarRating from './StarRating.vue'

interface Review {
  id: number
  title: string
  body: string
  rating: number
  approvedAt: string | null
}

const props = defineProps<{ review: Review; role: string }>()
const emit = defineEmits<{ (e: 'moderated', id: number): void }>()

const pending = ref(false)

const canModerate = computed(() => props.role === 'admin' && !props.review.approvedAt)

async function approve(): Promise<void> {
  pending.value = true
  try {
    await fetch(`/api/reviews/${props.review.id}/approve`, { method: 'POST' })
    emit('moderated', props.review.id)
  } finally {
    pending.value = false
  }
}

async function reject(): Promise<void> {
  pending.value = true
  try {
    await fetch(`/api/reviews/${props.review.id}/reject`, { method: 'POST' })
    emit('moderated', props.review.id)
  } finally {
    pending.value = false
  }
}
</script>

<style scoped>
.review-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
}

.review-card__header {
  display: flex;
  justify-content: space-between;
}
</style>
