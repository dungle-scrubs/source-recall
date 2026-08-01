@extends('layouts.app')

{{-- The @section directive inside this comment must not open a block. --}}

@section('title', 'Reviews')

@section('content')
    <div class="reviews">
        <x-page-heading :title="__('Reviews')" badge="{{ $reviews->total() }}" />

        @if ($reviews->isEmpty())
            <x-empty-state message="No reviews yet." />
        @else
            <ul class="review-list">
                @foreach ($reviews as $review)
                    <li class="review-list__item" data-id="{{ $review->id }}">
                        <x-review-card :review="$review">
                            @slot('footer')
                                <span class="rating">{{ $review->rating }}/5</span>
                            @endslot
                        </x-review-card>
                    </li>
                @endforeach
            </ul>

            {{ $reviews->links() }}
        @endif
    </div>
@endsection

@push('scripts')
    <script>
        window.reviewFilters = @json($filters);
    </script>
@endpush

@php
    $footerNote = trans('reviews.footer');
@endphp

<footer class="reviews-footer">@@notADirective {{ $footerNote }}</footer>
