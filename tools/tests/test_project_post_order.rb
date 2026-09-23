# frozen_string_literal: true

require 'liquid'
require_relative '../../_plugins/project-post-order'

class ProjectPostOrderTest
  include ProjectPostOrder

  def assert_equal(expected, actual)
    raise "Expected #{expected.inspect}, got #{actual.inspect}" unless expected == actual
  end

  def post(title, category = '2-5. SeSAC-Note', date = '2026-09-24')
    { 'title' => title, 'categories' => ['2.PROJECT', category], 'date' => date, 'url' => "/#{title}/" }
  end

  def test_numeric_series_order_without_mutating_input
    input = [post('10. Ten'), post('2. Two'), post('01. One')]
    assert_equal ['01. One', '2. Two', '10. Ten'], project_posts_in_order(input).map { |p| p['title'] }
    assert_equal '10. Ten', input.first['title']
  end

  def test_parent_category_groups_projects_before_series
    input = [post('01. B', '2-10. B'), post('02. A', '2-9. A'), post('01. A', '2-9. A')]
    assert_equal ['01. A', '02. A', '01. B'], project_posts_in_order(input).map { |p| p['title'] }
  end

  def test_unnumbered_posts_follow_numbered_series
    assert_equal ['01. Intro', 'Parser'], project_posts_in_order([post('Parser'), post('01. Intro')]).map { |p| p['title'] }
  end

  def test_non_project_and_mixed_lists_keep_their_order
    study = { 'title' => 'Study', 'categories' => ['3.STUDY'], 'date' => '2025-01-01' }
    input = [study, post('01. Intro')]
    assert_equal input, project_posts_in_order(input)
    assert_equal [study], project_posts_in_order([study])
  end

  def test_empty_and_single_item_lists
    assert_equal [], project_posts_in_order(nil)
    assert_equal [], project_posts_in_order([])
    assert_equal [post('4.5HZ')], project_posts_in_order([post('4.5HZ')])
  end

  def test_duplicate_numbers_are_deterministic
    input = [post('01. B'), post('01. A')]
    assert_equal project_posts_in_order(input), project_posts_in_order(input.reverse)
  end
end

tests = ProjectPostOrderTest.new
methods = tests.public_methods.grep(/^test_/).sort
methods.each { |method| tests.public_send(method) }
puts "#{methods.size} project ordering tests passed"
