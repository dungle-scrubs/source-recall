<template>
  <section class="stats-panel">
    <h2>{{ title }}</h2>
    <dl>
      <div v-for="stat in stats" :key="stat.key">
        <dt>{{ stat.label }}</dt>
        <dd>{{ formatValue(stat.value) }}</dd>
      </div>
    </dl>
  </section>
</template>

<script>
import { formatCurrency } from '../lib/format'

export default {
    name: 'stats-panel',
    props: {
        shopId: { type: Number, required: true },
    },
    components: { },
    data() {
        return {
            stats: [],
            loading: false,
        }
    },
    computed: {
        title() {
            return this.loading ? 'Loading…' : 'Shop stats'
        },
    },
    methods: {
        async fetchStats() {
            this.loading = true
            try {
                const res = await fetch(`/api/shops/${this.shopId}/stats`)
                this.stats = await res.json()
            } finally {
                this.loading = false
            }
        },
        formatValue(value) {
            return formatCurrency(value)
        },
    },
    mounted() {
        this.fetchStats()
    },
}
</script>
